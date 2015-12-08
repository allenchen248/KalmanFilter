import csv

def read_csv(fname):
	with open(fname, 'r') as csvfile:
		freader = csv.reader(csvfile, delimiter=',')
		headers = freader.next()
		rows = [row for row in freader]
		cols = zip(*rows)
		return {headers[i]:cols[i] for i in xrange(len(cols))}

