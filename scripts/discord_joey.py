# coding: utf-8
"""
JoeyNMT Discord Bot

You need to modify the constants `TOKEN`, `guild` and `CFG_FILES` at the beginning of
the script to add your bot's access token and the pointers to your joeynmt config files.

To get your access token, go
https://discord.com/developers/applications -> Bot -> Token

cf.)
- Install py-cord:
    https://guide.pycord.dev/installation
    :Warning: Apparently, there are some problems with the latest verison.
              Please use py-cord v2.0.1, instead.
    ```
    $ pip install --upgrade git+https://github.com/Pycord-Development/pycord@v2.0.1
    ```
- Creating Your First Bot:
    https://guide.pycord.dev/getting-started/creating-your-first-bot
- Slash Commands:
    https://guide.pycord.dev/interactions/application-commands/slash-commands
"""
from functools import partial
from pathlib import Path

import discord
from discord import SlashCommandGroup, option

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

TOKEN = "your-bot-token-here"  # replace with your bot token
guild = 123456789  # replace with your guild ID

# path to config files
CFG_FILES = {
    'ende': './wmt14_ende/config.yaml',
    'deen': './wmt14_deen/config.yaml',
}

# pycord bot client
bot = discord.Bot(debug_guilds=[guild])
translate = SlashCommandGroup("translate", "JoeyNMT translates a message.")


def prepare(cfg_file):
    print("Discord Joey: Loading a model ...")

    cfg = load_config(Path(cfg_file))
    # parse and validate cfg
    model_dir, load_model, device, n_gpu, _, _, fp16 = parse_train_args(
        cfg["training"], mode="prediction")

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

    print(f"\t{src_lang}-{trg_lang} model loaded successfully!")
    return test_data, model, test_cfg, device, n_gpu, fp16


def generate(src_input, lang_tag):
    bot.data_dict[lang_tag].cache = {}  # reset cache
    bot.data_dict[lang_tag].set_item(src_input)
    _, _, translations, _, _, _ = predict(
        model=bot.model_dict[lang_tag],
        data=bot.data_dict[lang_tag],
        compute_loss=False,
        device=bot.device[lang_tag],
        n_gpu=bot.n_gpu[lang_tag],
        normalization="none",
        num_workers=0,
        cfg=bot.cfg_dict[lang_tag],
        fp16=bot.fp16[lang_tag],
    )
    bot.data_dict[lang_tag].cache = {}  # reset cache
    return translations[0]


@bot.event
async def on_ready():
    print(f"Discord Joey: {bot.user} logged in.")

    for lang_tag, cfg_file in CFG_FILES.items():
        if lang_tag not in bot.model_dict:
            test_data, model, test_cfg, device, n_gpu, fp16 = prepare(cfg_file)
            bot.data_dict[lang_tag] = test_data
            bot.model_dict[lang_tag] = model
            bot.cfg_dict[lang_tag] = test_cfg
            bot.device[lang_tag] = device
            bot.n_gpu[lang_tag] = n_gpu
            bot.fp16[lang_tag] = fp16

    print("Discord Joey: ready to go!")
    print("=" * 20)


@translate.command(description="Translates a message from English to German.")
@option(
    "message",
    description="Enter a message to tranlsrate.",
    required=True,
    default="Hello!",
)
async def ende(ctx, message: str):
    assert 'ende' in bot.model_dict

    print(f"{ctx.author}: {message}")
    try:
        translation = generate(message.strip(), 'ende')
    except Exception as e:  # pylint: disable=broad-except
        translation = e
    print(f"{bot.user}: {translation}")
    print("=" * 20)

    # Note: `ctx.respond()` will cause an error for non-ascii string.
    # a workaound for now: Use `ctx.send()` instead.
    await ctx.send(f"{ctx.author}: {message}\n{bot.user}: {translation}")


@translate.command(description="Translates a message from German to English.")
@option(
    "message",
    description="Enter a message to tranlsrate.",
    required=True,
    default="Hallo!",
)
async def deen(ctx, message: str):
    assert 'deen' in bot.model_dict

    print(f"{ctx.author}: {message}")
    try:
        translation = generate(message.strip(), 'deen')
    except Exception as e:  # pylint: disable=broad-except
        translation = e
    print(f"{bot.user}: {translation}")
    print("=" * 20)

    # Note: `ctx.respond()` will cause an error for non-ascii string.
    # a work-around fo now: Use `ctx.send()` instead.
    await ctx.send(f"{ctx.author}: {message}\n{bot.user}: {translation}")


# manually add the slash command group created above
bot.add_application_command(translate)

# global variables
bot.data_dict = {}
bot.model_dict = {}
bot.cfg_dict = {}
bot.device = {}
bot.n_gpu = {}
bot.fp16 = {}

bot.run(TOKEN)
