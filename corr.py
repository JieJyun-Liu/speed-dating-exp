
import math
import collections

def dataFromFile(fname):

    with open(fname, 'rb') as f:
        data = [row for row in f.read().splitlines()]
    
    return data


if __name__ == "__main__":

	inFile = dataFromFile("interest.csv")

	
	# Attribute names
	attr_name = []

	# Collection of iid
	id_list = []
	id_set = set()

	inter_dict = {}

	for tid, instance in enumerate(inFile):
		inst_split = instance.split(",")

		if tid == 0:
			attr_name = inst_split
		else:
			id_set.add(inst_split[0])
	
	id_list = sorted(id_set)
	inter_dict = inter_dict.fromkeys(id_list)
	# for key in inter_dict.keys():
	# 	print key

	# print(inter_dict)
	error_list = []

	for tid, instance in enumerate(inFile):
		inst_split = instance.split(",")
		
		if tid != 0:
			key = inst_split[0]

			try: 
				inter_dict[key] = map(int, inst_split[4:])
			except:
				error_list.append(tid)

	# od = collections.OrderedDict(sorted(inter_dict.items))

	outfileName = "corr_result.csv"

	with open(outfileName, 'wt') as fout:

		for key_i in inter_dict.keys():
			for key_j in inter_dict.keys():
				
				if key_i == key_j:
					print key_i, " and ", key_j, " are the same."
					continue
				if (key_i in error_list) or (key_j in error_list):
					continue
				if (not inter_dict[key_i]) or (not inter_dict[key_j]):
					continue

				# print "inter_dict key i", key_i, inter_dict[key_i]
				# print "inter_dict key j", key_j, inter_dict[key_j]
				if not inter_dict[key_j]:
					print key_j, " is none."

				x = map(int, inter_dict[key_i][4:])
				y = map(int, inter_dict[key_j][4:])

				diffx = diffy = []
				xsum = ysum = 0.0
				xavg = yavg = 0.0

				for xi in x:
					xsum = xsum + xi

				xavg = xsum / len(x)

				for xi in x:
					diffx.append(xi-xavg)

				for yi in y:
					ysum = ysum + yi

				yavg = ysum / len(y)

				for yi in y:
					diffy.append(yi-yavg)

				# print diffx, diffy

				sum_diff = 0.00
				msum_diffx = 0.00
				msum_diffy = 0.00

				for i, xi in enumerate(x):
					sum_diff = diffx[i]*diffy[i]
					msum_diffx = msum_diffx + pow(diffx[i],2)
					msum_diffy = msum_diffy + pow(diffy[i],2) 

				prod_diff = msum_diffx * msum_diffy
				prod_diff = math.sqrt(prod_diff)

				rxy = sum_diff / prod_diff

				print key_i, " and ", key_j ," is ", rxy
				
				attr_instance = str(key_i) + "," + str(key_j) + "," + str(rxy) + '\n'
				fout.write(attr_instance)

	fout.close()




