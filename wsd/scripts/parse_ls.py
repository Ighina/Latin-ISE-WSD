import sys
from bs4 import BeautifulSoup
import pandas as pd
import re

def proc(filename):
	file=open(filename)

	page=""
	for line in file.readlines():  
		page+=line.rstrip() + " "

	file.close()
	soup=BeautifulSoup(page, features="lxml")

	entries=soup.findAll("entryfree", {"type":"main"})
	for entry in entries:
		key=entry["key"]
		orth=entry.findAll(["orth"], {"extent":"full"})[0].text

		senses=entry.findAll(["sense"])
		currentLevel1=None
		for sense in senses:
			n=sense["n"]
			level=int(sense["level"])

			if n == "I" or n == "II" or n == "III" or n == "IV" or n == "V":
				currentLevel1=n
			cites=sense.findAll(["cit"])
			for cite in cites:
				quotes=cite.findAll("quote", {"lang":"la"})
				bibl=None
				try:
					bibl=cite.findAll("bibl")[0].text
					author=cite.findAll("author")[0].text
					try:
					    code = cite.findAll("bibl")[0]["n"]
					except KeyError:
					    code = "None"
					sense="; ".join([x.get_text() for x in sense.findAll("hi")])
				except:
					pass
				for quote in quotes:
					text=quote.text
					print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (orth, key, currentLevel1, n, level, text, bibl, sense, author, code))

def add_info(filename, csv_file):
	"""
	Add info to the prediction of the wsd system to allow distant reading
	"""
	file=open(filename)

	page=""
	for line in file.readlines():
		page+=line.rstrip() + " "

	file.close()
	soup=BeautifulSoup(page, "html.parser")

	csv = pd.read_csv(csv_file)
	authors = []
	author_perseus_codes = []
	works = []
	senses = []
	for idx, sentence in enumerate(csv["sentence"]):
		x = soup.find("quote", text=sentence+",").parent
		author = x.find("author").get_text()
		work = re.sub(author+" ", "", x.find("bibl").get_text())
		code = x.find("bibl")["n"]
		sense_latin = csv["sense"][idx]
		lemma = csv["lemma"][idx]
		sense = soup.find_all(attrs={"key":lemma})[0].find_all(attrs={"level":"1", "n":sense_latin})[0].get_text()

		authors.append(author)
		author_perseus_codes.append(code)
		works.append(work)
		senses.append(sense)

	csv["authors"] = authors
	csv["author_codes"] = author_perseus_codes
	csv["works"] = works
	csv["english_sense"] = senses

	return csv


proc(sys.argv[1])

