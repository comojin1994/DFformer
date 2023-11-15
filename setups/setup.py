import os

def main():
    os.makedirs('../databases', exist_ok=True)
    os.makedirs('../checkpoints/bci2a', exist_ok=True)
    os.makedirs('../checkpoints/bci2b', exist_ok=True)
    os.makedirs('../checkpoints/sleepedf', exist_ok=True)
    os.makedirs('../checkpoints/shhs', exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
if __name__ == '__main__':
    main()