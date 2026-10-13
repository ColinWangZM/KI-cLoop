import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from application_retrieval import retrieve_application_routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question')
    parser.add_argument('--graph', type=Path, default=ROOT / 'outputs/kinetic_graph/graph.sqlite')
    args = parser.parse_args()
    result = retrieve_application_routes(args.question)
    if result is None:
        from graph_retrieval import Evidence
        with Evidence(args.graph) as evidence:
            result = evidence.retrieve(args.question)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
