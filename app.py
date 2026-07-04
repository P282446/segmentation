import argparse
from media.pipelines import full_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_directory",
                        help="Path to the output directory")
    parser.add_argument("--start-stage",
                        type=int,
                        default=0,
                        help="Start stage (default: 0)")
    parser.add_argument("--end-stage",
                        type=int,
                        default=6,
                        help="End stage (default: 6)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    final_path =  full_pipeline(args.input_file,
                  args.output_directory,
                  start_stage=args.start_stage,
                  end_stage=args.end_stage)
    print(f"Vidéo finale enregistrée : {final_path}")


if __name__ == "__main__":
    main()
