#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
import sys
import logging
import getopt
from aiohttp import web
from opencui.inference.converter import load_converter


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()

# This can be used to serve the whole thing, or just prompt service.


@routes.get("/")
async def hello(_: web.Request):  # For heart beat
    return web.Response(text="Hello, world")


@routes.post("/understand")
async def understand(request: web.Request):
    req = await request.json()
    logging.info(req)

    utterance = req.get("utterance")

    if len(utterance) == 0:
        return web.json_response({"errMsg": f'empty user input.'})

    l_converter = req.app["converter"]

    # So that we can use different llm.
    resp = l_converter.generate(utterance)
    return web.json_response(dataclasses.asdict(resp))


def init_app(converter):
    app = web.Application()
    app.add_routes(routes)
    app['converter'] = converter
    return app


if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    for opt, arg in opts:
        if opt == "-h":    
            print('serve.py -s <services/agent meta directory, separated by ,> -i <directory for index>')
            sys.exit()
        elif opt == "-i":
            index_path = arg
        elif opt == "-s":
            module_paths = arg

    # First load the schema info.
    converter = load_converter(module_paths, index_path)

    web.run_app(init_app(converter))
