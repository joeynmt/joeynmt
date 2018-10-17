import torch
import sacrebleu

from joeynmt.constants import PAD_TOKEN
from joeynmt.helpers import load_data, arrays_to_sentences, bpe_postprocess, \
    load_config, get_latest_checkpoint, make_data_iter, \
    load_model_from_checkpoint, store_attention_plots
from joeynmt.model import build_model
from joeynmt.batch import Batch


def validate_on_data(model, data, batch_size, use_cuda, max_output_length,
                     level, eval_metric, criterion, beam_size=0, beam_alpha=-1):
    """
    Generate translations for the given data.
    If `criterion` is not None and references are given, also compute the loss.
    :param model:
    :param data:
    :param batch_size:
    :param use_cuda:
    :param max_output_length:
    :param level:
    :param eval_metric:
    :param criterion:
    :param beam_size:
    :param beam_alpha:
    :return:
    """
    valid_iter = make_data_iter(dataset=data, batch_size=batch_size,
                          shuffle=False, train=False)
    valid_sources_raw = [s for s in data.src]
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        for valid_i, valid_batch in enumerate(iter(valid_iter), 1):
            # run as during training to get validation loss (e.g. xent)

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_lengths()

            # TODO save computation: forward pass is computed twice
            # run as during training with teacher forcing
            if criterion is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, criterion=criterion)
                total_loss += batch_loss
                total_ntokens += batch.ntokens

            # run as during inference to produce translations
            output, attention_scores = model.run_batch(
                batch=batch, beam_size=beam_size, beam_alpha=beam_alpha,
                max_output_length=max_output_length)

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None else [])

        assert len(all_outputs) == len(data)

        if criterion is not None and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoded_valid = arrays_to_sentences(arrays=all_outputs,
                                            vocabulary=model.trg_vocab,
                                            cut_at_eos=True)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe":
            valid_sources = [bpe_postprocess(s) for s in valid_sources]
            valid_references = [bpe_postprocess(v)
                                for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v) for
                                v in valid_hypotheses]

        # if references are given, evaluate against them
        if len(valid_references) > 0:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = sacrebleu.raw_corpus_bleu(
                    sys_stream=valid_hypotheses,
                    ref_streams=[valid_references]).score
            elif eval_metric.lower() == 'chrf':
                current_valid_score = sacrebleu.corpus_chrf(
                    hypotheses=valid_hypotheses,
                    references=valid_references)
        else:
            current_valid_score = -1

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
           valid_sources_raw, valid_references, valid_hypotheses, \
           decoded_valid, \
           valid_attention_scores


def test(cfg_file,
         ckpt: str = None,
         output_path: str = None,
         save_attention: bool = False):
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.
    :param cfg_file:
    :param ckpt:
    :param output_path:
    :param save_attention:
    :return:
    """

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(dir)
        try:
            step = ckpt.split(dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    batch_size = cfg["training"]["batch_size"]
    use_cuda = cfg["training"]["use_cuda"]
    level = cfg["data"]["level"]
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    # TODO load only test data
    train_data, dev_data, test_data, src_vocab, trg_vocab = \
        load_data(cfg=cfg)

    # TODO specify this differently
    data_to_predict = {"dev": dev_data, "test": test_data}

    # load model state from disk
    model_state = load_model_from_checkpoint(ckpt)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_state)

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 0
        beam_alpha = -1

    for data_set_name, data_set in data_to_predict.items():

        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=data_set, batch_size=batch_size, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, criterion=None, beam_size=beam_size,
            beam_alpha=beam_alpha)

        if "trg" in data_set.fields:
            decoding_description = "Greedy decoding" if beam_size == 0 else \
                "Beam search decoding with beam size = {} and alpha = {}".format(
                    beam_size, beam_alpha)
            print("{:4s} {}: {} [{}]".format(
                data_set_name, eval_metric, score, decoding_description))
        else:
            print("No references given for {} -> no evaluation.".format(
                data_set_name))

        if attention_scores is not None and save_attention:
            attention_path = "{}/{}.{}.att".format(dir, data_set_name, step)
            print("Attention plots saved to: {}.xx".format(attention_path))
            store_attention_plots(attentions=attention_scores,
                                  targets=hypotheses_raw,
                                  sources=[s for s in data_set.src],
                                  idx=range(len(hypotheses)),
                                  output_prefix=attention_path)

        if output_path is not None:
            output_path_set = "{}.{}".format(output_path, data_set_name)
            with open(output_path_set, mode="w", encoding="utf-8") as f:
                for h in hypotheses:
                    f.write(h + "\n")
            print("Translations saved to: {}".format(output_path_set))