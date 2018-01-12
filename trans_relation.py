def dataFromFile(fname):

    with open(fname, 'rb') as f:
        data = [row for row in f.read().splitlines()]
    
    return data


if __name__ == "__main__":

	inFile = dataFromFile("relation.csv")

	outFileName = "corr_dec.csv"

	# with open(outFileName, 'wt') as fout:
	with open(outFileName, 'wt') as fout:

		fout.write("int_corr,dec\n")

		for ind, f in enumerate(inFile):
			if ind == 0:
				continue
			instance = f.split(',')

			dec_exp = int(instance[3])*2 + int(instance[4])*1
			out_str = instance[2] + "," + str(dec_exp) + '\n'

			print out_str
			fout.write(out_str)
		# fout.write(final_rule)



