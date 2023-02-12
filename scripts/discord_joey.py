# coding: utf-8
"""
JoeyNMT Discord Bot

You need to modify the constants `TOKEN`, `guild` and `models` at the beginning of
the script to add your bot's access token and the joeynmt model names.

To get your access token, go
https://discord.com/developers/applications -> Bot -> Token
cf.)
- Install py-cord:
    https://guide.pycord.dev/installation
    ```
    $ pip install py-cord>=2.3.2
    ```
- Creating Your First Bot:
    https://guide.pycord.dev/getting-started/creating-your-first-bot
- Slash Commands:
    https://guide.pycord.dev/interactions/application-commands/slash-commands
"""

import discord
import torch
from discord import SlashCommandGroup, option

TOKEN = "your-bot-token-here"  # replace with your bot token
guild = 123456789  # replace with your guild ID

# model names
models = {'ende': 'wmt14_ende', 'deen': 'wmt14_deen'}

# pycord bot client
bot = discord.Bot(debug_guilds=[guild])
translate = SlashCommandGroup("translate", "JoeyNMT translates a message.")


@bot.event
async def on_ready():
    print(f"Discord Joey: {bot.user} logged in.")

    for lang_tag, model_name in models.items():
        if lang_tag not in bot.models:
            print("Discord Joey: Loading a model ...")
            bot.models[lang_tag] = torch.hub.load('joeynmt/joeynmt', model_name)
            print(f"\t{model_name} model loaded successfully!")

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
    assert 'ende' in bot.models

    print(f"{ctx.author}: {message}")
    try:
        translation = bot.models['ende'].translate([message])[0]
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
    assert 'deen' in bot.models

    print(f"{ctx.author}: {message}")
    try:
        translation = bot.models['deen'].translate([message])[0]
    except Exception as e:  # pylint: disable=broad-except
        translation = e
    print(f"{bot.user}: {translation}")
    print("=" * 20)

    # Note: `ctx.respond()` will cause an error for non-ascii string.
    # a work-around fo now: Use `ctx.send()` instead.
    await ctx.send(f"{ctx.author}: {message}\n{bot.user}: {translation}")


# manually add the slash command group created above
bot.add_application_command(translate)

# global variable
bot.models = {}

bot.run(TOKEN)
