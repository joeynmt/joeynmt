#!/usr/bin/env python
# coding: utf-8
"""
Normalize Japanese texts

Usage:
$ python sanitize.py < input.txt > output.txt

cf.
https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
http://lotus.kuee.kyoto-u.ac.jp/WAT/Timely_Disclosure_Documents_Corpus/specifications.html
"""
import re
import sys
import unicodedata


def _unicode_normalize(cls, s):
    # pylint: disable=consider-using-f-string
    pt = re.compile("([{}]+)".format(cls))

    def _norm(c):
        return unicodedata.normalize("NFKC", c) if pt.match(c) else c

    s = "".join(_norm(x) for x in re.split(pt, s))
    return s


def _remove_extra_spaces(s):
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)
    blocks = "".join((
        "\u4E00-\u9FFF",  # CJK UNIFIED IDEOGRAPHS
        "\u3040-\u309F",  # HIRAGANA
        "\u30A0-\u30FF",  # KATAKANA
        "\u3000-\u303F",  # CJK SYMBOLS AND PUNCTUATION
        "\uFF00-\uFFEF",  # HALFWIDTH AND FULLWIDTH FORMS
    ))

    # latin = ''.join(('\u0000-\u007F',   # Basic Latin[g]
    #                 '\u0080-\u00FF',   # Latin-1 Supplement[h]
    # ))

    def _remove_space_between(cls1, cls2, s):
        # pylint: disable=consider-using-f-string
        p = re.compile("([{}]) ([{}])".format(cls1, cls2))
        while p.search(s):
            s = p.sub(r"\1\2", s)
        return s

    s = _remove_space_between(blocks, blocks, s)
    # s = _remove_space_between(blocks, latin, s)
    # s = _remove_space_between(latin, blocks, s)
    return s


def normalize(s):
    s = s.strip()
    s = re.sub("\t", " ", s)
    s = _unicode_normalize("０-９Ａ-Ｚａ-ｚ｡-ﾟ", s)

    def _maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub("[˗֊‐‑‒–⁃⁻₋−]+", "-", s)  # normalize hyphens
    s = re.sub("[﹣－ｰ—―─━ー]+", "ー", s)  # normalize choonpus
    s = re.sub("[~∼∾〜〰～]+", "〜", s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        _maketrans(
            "!\"#$%&'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣",
            "！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」",
        ))

    s = _remove_extra_spaces(s)
    s = _unicode_normalize("！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜", s)  # keep ＝,・,「,」
    s = re.sub("[’]", "'", s)
    s = re.sub("[”]", '"', s)
    s = re.sub("[“]", '"', s)
    return s


if __name__ == "__main__":
    for line in sys.stdin.readlines():
        line = line.rstrip("\n")
        print(normalize(line))
