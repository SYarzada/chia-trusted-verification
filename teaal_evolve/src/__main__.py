import sys                    

from src.ir.program import *                    
from src.trans import *                    


def main():                    
                                                  
    if len(sys.argv) < 3:
        print("Usage: python -m src [mode] [input files]")
        return

    mode = sys.argv[1]

    if mode == "check":
        NKIMain.check(files=sys.argv[2:])

               
    elif mode == "nki":
        print(NKIMain.compile(files=sys.argv[2:]))

                             
    elif mode == "teaal":
        TeAALMain.generate_from_einsum(sys.argv[2], files=sys.argv[3:])

                                   
    elif mode == "match":
        ModelMain.match_all(sys.argv[2], files=sys.argv[3:])

    else:
        print("Unknown mode: " + mode)
        print("Usage: python -m src [mode] [files]")


if __name__ == "__main__":                    
    main()
