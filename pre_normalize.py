
def dataFromFile(fname):

    with open(fname, 'rb') as f:
        data = [row for row in f.read().splitlines()]
    
    return data

if __name__ == "__main__":

	inFile = dataFromFile("preference_o.csv")

	outFileName = "preference_o_trans.csv"
	with open(outFileName, 'w') as fout:

		for ind, f in enumerate(inFile):
			inst = f.split(",")

			outStr = ""
			for iid, item in enumerate(inst):
				if item == '':
					item = -1

				if iid > 0:
					outStr += str(item) + ","

			outStr = outStr[:-1]
			outStr += '\n'	
			print outStr
			fout.write(outStr)

	fout.close()







