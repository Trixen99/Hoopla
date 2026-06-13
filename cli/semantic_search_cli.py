#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk_text, semantic_chunk, ChunkedSemanticSearch
import os, json

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify language Model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Encode a text string")
    embed_text_parser.add_argument("text", type=str, help="String to encode using the Text Embedder")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="locate or create embeddings for provided document file")

    embed_query_parser = subparsers.add_parser("embed_query", help="convert user query into embedding")
    embed_query_parser.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="search the database")
    search_parser.add_argument("text", type=str, help="String to search")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Tunable limit variable (how many movies would you like to return)")



    chunk_parser = subparsers.add_parser("chunk", help="split long text into smaller pieces for embedding")
    chunk_parser.add_argument("text", type=str, help="String to to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="chunk size to use")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="overlap buffer to use")



    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="split text at natural breaks (ie sentence ends and end of paragraphs)")
    semantic_chunk_parser.add_argument("text", type=str, help="String to to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="max size of chunks generated")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="how much overlap to use")


    semantic_chunk_embed_chunks_parser = subparsers.add_parser("embed_chunks", help="embed the provided documents")


    search_chunked_parser = subparsers.add_parser("search_chunked", help="search the chunked movie database using the provided query text")
    search_chunked_parser.add_argument("text", type=str, help="String to search")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="limit")

    args = parser.parse_args()
        


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.text)

        case "search":
            search(args.text, args.limit)

        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks,1):
                print(f"{i}. {chunk}")
            
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
            return          

        case "embed_chunks":
            with open(os.path.abspath('data/movies.json')) as moviedata:
                documents = json.load(moviedata)["movies"]  
            model = ChunkedSemanticSearch()
            embeddings = model.load_or_create_chunk_embeddings(documents)
            print(len(documents))
            print(f"Generated {len(embeddings)} chunked embeddings")



        case "search_chunked":
            with open(os.path.abspath('data/movies.json')) as moviedata:
                documents = json.load(moviedata)["movies"]  
            model = ChunkedSemanticSearch()
            embeddings = model.load_or_create_chunk_embeddings(documents)
            results = model.search_chunks(args.text, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"   {result["document"]}...")




        case _:
            parser.print_help()

        



if __name__ == "__main__":
    main()