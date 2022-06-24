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

client = discord.Client()

# access token
# Go https://discord.com/developers/applications -> Bot -> Token
TOKEN = 'your-access-token-here'

CFG_FILES = {
    'en-ja': './models/jparacrawl_enja/config.yaml',
    'ja-en': './models/jparacrawl_jaen/config.yaml'
}
DEVICE = torch.device("cuda")  # DEVICE = torch.device("cpu")
N_GPU = 1  # N_GPU = 0


def load_joeynmt_model(cfg_file):
    cfg = load_config(Path(cfg_file))
    # parse and validate cfg
    model_dir, load_model, device, n_gpu, _, _ = parse_train_args(cfg["training"],
                                                                  mode="prediction")
    assert device.type == DEVICE.type
    assert n_gpu == N_GPU

    # read vocabs
    src_vocab, trg_vocab = build_vocab(cfg["data"], model_dir=model_dir)

    # build model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from disk
    ckpt = resolve_ckpt_path(None, load_model, model_dir)
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

    # override decoding options
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
    assert lang_tag in CFG_FILES

    src_input = src_input[7:]
    src_input = src_input.strip()

    # remove emojis
    emoji_pattern = re.compile(r"\:[a-zA-Z]+\:")
    src_input = re.sub(emoji_pattern, "", src_input)
    src_input = src_input.strip()
    assert src_input is not None and len(src_input) > 0

    print(f"/{lang_tag}/", src_input)  # print console log
    return lang_tag, src_input


def translate(src_input, model, test_data, cfg):
    test_data.cache = {}  # reset cache
    test_data.set_item(src_input)
    _, _, translations, _, _, _ = predict(
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
    return translations[0]


@client.event
async def on_ready():
    print('Logged in.')

    global data_dict, model_dict, cfg_dict  # pylint: disable=global-variable-undefined
    data_dict = {}
    model_dict = {}
    cfg_dict = {}
    for lang_tag, cfg_file in CFG_FILES.items():
        test_data, model, test_cfg = load_joeynmt_model(cfg_file)
        data_dict[lang_tag] = test_data
        model_dict[lang_tag] = model
        cfg_dict[lang_tag] = test_cfg

    print('=' * 20)  # ready to go!


@client.event
async def on_message(message):
    # ignore, if a bot throws a message
    if message.author.bot:
        return

    # get source input
    src_input = message.content.strip()
    lang_tag, src_input = get_language_tag(src_input)

    if lang_tag in CFG_FILES:
        # get translation
        translation = translate(
            src_input,
            model_dict[lang_tag],
            data_dict[lang_tag],
            cfg_dict[lang_tag],
        )
        print(f'JoeyNMT: {translation}\n')  # print console log

        # return translation
        await message.channel.send(translation)


client.run(TOKEN)
