import argparse
import os



def main():
    parser = argparse.ArgumentParser(description="Define and run the experiment with one config")
    parser.add_argument('--model', type=str, default="Small", help="")
    parser.add_argument('--loss', type=str, default="mse", help="")
    parser.add_argument('--observables', type=int, help="")



    arguments = vars(parser.parse_args())

    if arguments['model'] == 'QML':
        script ='train_QML.py'
        for i in range(1,11):
            os.system(f"python {script} --run {i} --loss {arguments['loss']} --observables {arguments['observables']}")
    else:
        script ='train_classic.py'
        for i in range(1,11):
            os.system(f"python {script} --run {i} --model {arguments['model']}")


if __name__ == "__main__":
    main()

