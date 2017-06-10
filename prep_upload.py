import sys
from pathlib import Path

def main():
    ifname = sys.argv[1]
    _id = int(sys.argv[2])
    p = Path(ifname) 
    ofname = p.with_name(p.stem + '-upload' + p.suffix)
    with open(ifname) as inf, open(ofname, 'w') as outf:
        met = False
        for line in inf:
            line = line.rstrip('\n')
            try:
                int(line)
                met = False
            except ValueError:
                if not met:
                    line = line.split("@@ ")[1]
                    outf.write("%d %s" % (_id, line))
                    outf.write("\n")
                    print("%d %s" % (_id, line))
                    met = True

if __name__ == "__main__":
    main()
