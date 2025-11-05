def main():
    parser = argparse.ArgumentParser(description='Hadoop Word Count Mapper and Reducer')
    parser.add_argument('mode', choices=['map', 'reduce'], help='Choose to run the script as mapper or reducer')
    args = parser.parse_args()
    if args.mode == 'map':
        mapper()
    elif args.mode == 'reduce':
        reducer()

if __name__ == '__main__':
    main()
