"""CLI command runner for Ghana Chat."""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(prog="ghana-chat", description="Ghana Chat CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ask command
    ask_p = subparsers.add_parser("ask", help="Ask a question via CLI")
    ask_p.add_argument("query", type=str, help="Question to ask")
    ask_p.add_argument("--max-triples", type=int, default=8, help="Max triples to retrieve")

    # Serve command
    serve_p = subparsers.add_parser("serve", help="Run the API server")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_p.add_argument("--workers", type=int, default=1, help="Uvicorn workers")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run("ghana_chat.server:app", host=args.host, port=args.port, workers=args.workers)
    elif args.command == "ask" or (len(sys.argv) > 1 and not sys.argv[1].startswith("-")):
        query = args.query if args.command == "ask" else sys.argv[1]
        from .retriever import HeadKGRetriever
        from .generator import GroundedGenerator

        retriever = HeadKGRetriever()
        generator = GroundedGenerator()

        print(f"\nQuestion: {query}")
        ret = retriever.retrieve(query)
        triples = ret.get("triples", [])
        print(f"Retrieved {len(triples)} head facts.")
        for t in triples:
            print(f"  - {t['head']} | {t['relation']} | {t['tail']}")

        print("\nGenerating grounded answer...")
        res = generator.generate(query, triples)
        print(f"\nAnswer:\n{res['answer']}")
        print(f"\n[Generated {res['tokens_generated']} tokens in {res['latency_s']}s ({res['tokens_per_sec']} tok/s)]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
