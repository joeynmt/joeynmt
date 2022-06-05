# coding: utf-8
"""
JoeyNMT Discord Bot

cf.)
Install discord.py: https://discordpy.readthedocs.io/en/stable/intro.html#installing
A Minimal Bot: https://discordpy.readthedocs.io/en/stable/quickstart.html
Getting started: https://discord.com/developers/docs/getting-started
"""
from functools import partial
from pathlib import Path
import re

import discord
import torch

from joeynmt.datasets import build_dataset
from joeynmt.helpers import (
    load_checkpoint,
    load_config,
    parse_train_args,
    resolve_ckpt_path,
)
from joeynmt.model import build_model
from joeynmt.prediction import predict
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import build_vocab
# joeynmt v2.0.0

# access token
# Go https://discord.com/developers/applications -> Bot -> Token
TOKEN = 'your-access-token-here'

DEVICE = torch.device("cuda")  # DEVICE = torch.device("cpu")
N_GPU = 1  # N_GPU = 0

cfg_files = {
    'en-ja': './models/jparacrawl_enja/config.yaml',
    'ja-en': './models/jparacrawl_jaen/config.yaml'
}

client = discord.Client()


def load_joeynmt_model(cfg_file):
    cfg = load_config(Path(cfg_file))
    # parse and validate cfg
    model_dir, load_model, device, n_gpu, _, _ = parse_train_args(cfg["training"],
                                                                  mode="prediction")
    assert device.type == DEVICE.type
    assert n_gpu == N_GPU

    # when checkpoint is not specified, take latest (best) from model dir
    ckpt = resolve_ckpt_path(None, load_model, model_dir)

    # read vocabs
    src_vocab, trg_vocab = build_vocab(cfg["data"], model_dir=model_dir)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])

    if device.type == "cuda":
        model.to(device)

    src_lang = cfg["data"]["src"]["lang"]
    trg_lang = cfg["data"]["trg"]["lang"]
    tokenizer = build_tokenizer(cfg["data"])
    sequence_encoder = {
        src_lang: partial(src_vocab.sentences_to_ids, bos=False, eos=True),
        trg_lang: None,
    }
    test_data = build_dataset(
        dataset_type="stream",
        path=None,
        src_lang=src_lang,
        trg_lang=trg_lang,
        split="test",
        tokenizer=tokenizer,
        sequence_encoder=sequence_encoder,
    )

    test_cfg = cfg["testing"]
    test_cfg["batch_type"] = "sentence"
    test_cfg["batch_size"] = 1
    test_cfg["n_best"] = 1
    test_cfg["return_prob"] = "none"
    test_cfg["return_attention"] = False

    print(f"Joey NMT: {src_lang}-{trg_lang} model loaded successfully.")
    return test_data, model, test_cfg


def get_language_tag(src_input):
    lang_tag = src_input[:7].strip('/')
    assert lang_tag in cfg_files

    src_input = src_input[7:]
    src_input = src_input.strip()

    # remove emojis
    emoji_pattern = re.compile(r"\:[a-zA-Z]+\:")
    src_input = re.sub(emoji_pattern, "", src_input)
    src_input = src_input.strip()
    assert src_input is not None and len(src_input) > 0

    return lang_tag, src_input


def translate(src_input, model, test_data, cfg):
    test_data.cache = {}  # reset cache
    test_data.set_item(src_input)
    _, _, hypotheses, _, _, _ = predict(
        model=model,
        data=test_data,
        compute_loss=False,
        device=DEVICE,
        n_gpu=N_GPU,
        normalization="none",
        num_workers=0,
        cfg=cfg,
    )
    test_data.cache = {}  # reset cache
    return hypotheses[0]


@client.event
async def on_ready():
    # print console log
    print('logged in.')

    test_data_enja, model_enja, cfg_enja = load_joeynmt_model(cfg_files['en-ja'])
    test_data_jaen, model_jaen, cfg_jaen = load_joeynmt_model(cfg_files['ja-en'])

    global data_dict, model_dict, cfg_dict  # pylint: disable=global-variable-undefined
    data_dict = {'en-ja': test_data_enja, 'ja-en': test_data_jaen}
    model_dict = {'en-ja': model_enja, 'ja-en': model_jaen}
    cfg_dict = {'en-ja': cfg_enja, 'ja-en': cfg_jaen}


# message event
@client.event
async def on_message(message):
    # ignore, if a bot throws a message
    if message.author.bot:
        return

    # get source input
    src_input = (message.content).strip()
    if src_input.startswith('/en-ja/') or src_input.startswith('/ja-en/'):
        lang_tag, src_input = get_language_tag(src_input)
        print(lang_tag, src_input)  # print console log

        # get translation
        hypothesis = translate(src_input, model_dict[lang_tag], data_dict[lang_tag],
                               cfg_dict[lang_tag])
        print('JoeyNMT', hypothesis)  # print console log

        # return translation
        await message.channel.send(hypothesis)


client.run(TOKEN)
