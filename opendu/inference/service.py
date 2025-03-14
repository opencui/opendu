#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
import getopt
import logging
import sys
from enum import Enum
import os
from lru import LRU
import traceback as tb
from aiohttp import web
import shutil
from opendu.core.config import RauConfig
from opendu.inference.parser import Parser, Generator, load_parser
from opendu.core.index import indexing
from sentence_transformers import SentenceTransformer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()


Enum("DugMode", ["SKILL", "SLOT", "BINARY", "SEGMENT"])

# This service is designed to provide the indexing and serving for many agents, so that
# We can reduce the startup time.
"""
curl -X POST -d '{"mode":"SKILL","utterance":"make a reservation","expectations":[],"slotMetas":[],"entityValues":{},"questions":[]}' 127.0.0.1:3001/v1/predict/tableReservation
curl -X POST -d '{"mode":"BINARY","utterance":"Yes, absolutely.","questions":["Are you sure you want the white one?"]}' 127.0.0.1:3001/v1/predict/agent
curl -X POST -d '{"mode": "SLOT", "utterance": "order food", "slots": [], "candidates": {}, "dialogActs": []}' http://127.0.0.1:3001/v1/predict/tableReservation
"""

@routes.get("/hello")
async def hello(_: web.Request):  # For heart beat
    return web.Response(text=f"Ok")


@routes.get("/v1/index/{bot}")
async def index(request: web.Request):
    bot = request.match_info['bot']
    root = request.app["root"]
    bot_path = f"{root}/{bot}"
    index_path = f"{root}/{bot}/index"
    # We remove converter, delete the index, and index again.
    converters = request.app["converters"]

    # Remove the old object.
    converters[bot] = None
    if os.path.exists(index_path):
        logging.info(f"remove index for {bot}")
        shutil.rmtree(index_path)

    logging.info(f"create index for {bot}")
    try:
        indexing(bot_path)

        # Assume it is always a good idea to reload the index.
        reload(bot, request.app)
    except Exception as e:
        traceback_str = ''.join(tb.format_exception(None, e, e.__traceback__))
        return web.Response(text=traceback_str, status=500)

    return web.Response(text="Ok")


@routes.get("/v1/load/{bot}")
async def load(request: web.Request):
    bot = request.match_info['bot']
    try:
        reload(bot, request.app)
    except Exception as e:
        traceback_str = ''.join(tb.format_exception(None, e, e.__traceback__))
        return web.Response(text=traceback_str, status=500)

    # client will only check 200.
    return web.Response(text="Ok")


@routes.post("/v1/predict/{bot}")
async def understand(request: web.Request):
    bot = request.match_info['bot']

    # Make sure we have reload the index.
    try:
        reload(bot, request.app)
    except Exception as e:
        traceback_str = ''.join(tb.format_exception(None, e, e.__traceback__))
        return web.Response(text=traceback_str, status=500)

    req = await request.json()
    logging.info(req)

    utterance = req.get("utterance")

    if len(utterance) == 0:
        return web.json_response({"errMsg": f"empty user input."})

    mode = req.get("mode")
    l_converter: Parser = request.app["converters"][bot]

    if mode == "DESCSIM":
        descriptions = req.get("descriptions")
        return web.json_response({})

    if mode == "EXEMPLARSIM":
        exemplars = req.get("exemplars")
        return web.json_response({})

    if mode == "DEBUG":
        expectations = req.get("expectations")
        results = l_converter.debug(utterance, expectations)
        return web.json_response(results)

    if mode == "SEGMENT":
        return web.json_response({"errMsg": f"Not implemented yet."})

    if mode == "SKILL":
        try:
            expectations = req.get("expectations")
            candidates = req.get("candidates")
            results = l_converter.detect_triggerables(utterance, expectations)
        except Exception as e:
            traceback_str = ''.join(tb.format_exception(None, e, e.__traceback__))
            return web.Response(text=traceback_str, status=500)

        return web.json_response(results)

    if mode == "SLOT":
        try:
            slots = req.get("slots")
            entities = req.get("candidates")
            results = l_converter.fill_slots(utterance, slots, entities)
            logging.info(results)
        except Exception as e:
            traceback_str = ''.join(tb.format_exception(None, e, e.__traceback__))
            return web.Response(text=traceback_str, status=500)

        return web.json_response(results)

    if mode == "BINARY":
        try:
            questions = req.get("questions")
            dialog_acts = req.get("dialogActs")
            # So that we can use different llm.
            resp = l_converter.inference(utterance, questions)
        except Exception as e:
            traceback_str = ''.join(tb.format_exception(None, e, e.__traceback__))
            return web.Response(text=traceback_str, status=500)

        return web.json_response(resp)


# This reload the converter from current indexing.
def reload(key, app):
    root = app["root"]
    converters = app["converters"]
    bot_path = f"{root}/{key}"
    if key not in converters or converters[key] is None:
        logging.info(f"load index for {key}...")
        index_path = f"{bot_path}/index/"
        converters[key] = load_parser(bot_path, index_path)
        logging.info(f"bot {key} is ready.")


def init_app(schema_root, size):
    app = web.Application()
    app.add_routes(routes)
    app["converters"] = LRU(size)
    app['root'] = schema_root
    return app


if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    lru_capacity = 32
    for opt, arg in opts:
        if opt == "-h":
            print(
                "serve.py -s <root for services/agent schema>"
            )
            sys.exit()
        elif opt == "-s":
            root_path = arg
        elif opt == "-i":
            lru_capacity = int(arg)

    # This load the generator LLM first.
    embedder = SentenceTransformer(RauConfig.get().embedding_model, device=RauConfig.get().embedding_device, trust_remote_code=True)
    Generator.build()
    web.run_app(init_app(root_path, lru_capacity), port=3001)
