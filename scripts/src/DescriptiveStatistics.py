# -*- coding: utf-8 -*-
"""
@author: Adrián Bazaga
"""

import MySQLdb
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import pandas as pd
import _pickle as cPickle

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.02*float(h), '%d'%int(h),
                ha='center', va='bottom', rotation=90)


journal_colors = cPickle.load(open("journal_colors.pickle", "rb"))
topic_colors = cPickle.load(open("topic_colors.pickle", "rb"))
institution_colors = cPickle.load(open("institution_colors.pickle", "rb"))
cmap_tab20 = cPickle.load(open("cmap_tab20.pickle", "rb"))

years = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
journals = ["BMC Bioinformatics", "BMC Genomics", "Nucleic Acids Research", "Oxford Bioinformatics", "PLoS Computational Biology"]
#journal_colors = ['r', 'g', 'y', 'b', 'orange']
articles_per_year = [] #2005 -> 2017
articles_per_journal = []


# Open database connection
db = MySQLdb.connect("","","","" )

cursor = db.cursor()

# Number of articles per year
cursor.execute("SELECT COUNT(DISTINCT(title)) FROM - GROUP BY year")
for row in cursor:
    articles_per_year.append(row[0])
    
y_pos = np.arange(len(years))
bars = plt.bar(y_pos, articles_per_year, align='center', alpha=0.5)
plt.xticks(y_pos, years)
plt.xticks(rotation=45)
plt.ylabel('Number of articles')
plt.xlabel('Year')
plt.title('Number of published articles per year')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

 
plt.show()

# Number of citations per year
citations_per_year = []
cursor.execute("SELECT SUM(citations) FROM articlesmulti3topics2 WHERE year > 2004 GROUP BY year")
for row in cursor:
    citations_per_year.append(row[0])
    
y_pos = np.arange(len(years))
fig = plt.figure(figsize=(15, 13))
bars = plt.bar(y_pos, citations_per_year, align='center', alpha=0.5)
plt.xticks(y_pos, years)
plt.xticks(rotation=45)
plt.ylabel('Number of citations')
plt.xlabel('Year')
plt.title('Number of citations per year')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom', rotation=45)


plt.show()

# Number of articles per journal
cursor.execute("SELECT COUNT(DISTINCT(title)) FROM articlesmulti3topics2 WHERE year > 2004 GROUP BY source ORDER BY source ASC")
for row in cursor:
    articles_per_journal.append(row[0])
    
y_pos = np.arange(len(journals))
bars = plt.bar(y_pos, articles_per_journal, align='center', alpha=0.5, color=journal_colors)
plt.xticks(y_pos, journals)
plt.xticks(rotation=45)
plt.ylabel('Number of articles')
plt.xlabel('Journal')
plt.title('Number of published articles per journal')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

 
plt.show()

plt.pie(articles_per_journal, labels=journals, autopct='%1.0f%%', pctdistance=0.6, labeldistance=1.1, shadow=False)
plt.title("Percentage of articles per journal")
plt.show()

# Number of citations per journal
citations_per_journal = []
cursor.execute("SELECT SUM(citations) FROM articlesmulti3topics2 WHERE year > 2004 GROUP BY source ORDER BY source ASC")
for row in cursor:
    citations_per_journal.append(row[0])
    
y_pos = np.arange(len(journals))
bars = plt.bar(y_pos, citations_per_journal, align='center', alpha=0.5, color=journal_colors)
plt.xticks(y_pos, journals)
plt.xticks(rotation=45)
plt.ylabel('Number of articles')
plt.xlabel('Journal')
plt.title('Number of citations per journal')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

 
plt.show()

plt.pie(citations_per_journal, labels=journals, autopct='%1.0f%%', pctdistance=0.6, labeldistance=1.1, shadow=False)
plt.title("Percentage of citations per journal")
plt.show()

# Evolution of articles per journal
cursor.execute("SELECT COUNT(DISTINCT(title)) FROM articlesmulti3topics2 WHERE year > 2004 GROUP BY source,year ORDER BY source,year ASC")
articles_per_year_bmcbio = []
articles_per_year_bmcgen = []
articles_per_year_nar = []
articles_per_year_oxfbio = []
articles_per_year_ploscb = []

for index, row in enumerate(cursor):
    if(index < 13):
        articles_per_year_bmcbio.append(row[0])
    if(index > 12 and index < 26):
        articles_per_year_bmcgen.append(row[0])
    if(index > 25 and index < 39):
        articles_per_year_nar.append(row[0])
    if(index > 38 and index < 52):
        articles_per_year_oxfbio.append(row[0])
    if(index > 51):
        articles_per_year_ploscb.append(row[0])

# Evolution of articles per journal in bar plot

bar_width = 0.15
index = np.arange(len(journals))
y_stack = np.row_stack((articles_per_year_bmcbio,articles_per_year_bmcgen,articles_per_year_nar,articles_per_year_oxfbio,articles_per_year_ploscb))
colormap = plt.cm.gist_ncar 
y_pos = np.arange(len(years))
index = np.arange(13)
fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(111)
rects1 = plt.bar(index, y_stack[0,:], bar_width, label="BMC Bioinformatics", color=journal_colors[0])
rects2 = plt.bar(index + bar_width, y_stack[1,:], bar_width, label="BMC Genomics", color=journal_colors[1])
rects3 = plt.bar(index + (bar_width*2), y_stack[2,:], bar_width, label="Nucleic Acids Research", color=journal_colors[2])
rects4 = plt.bar(index + (bar_width*3), y_stack[3,:], bar_width, label="Oxford Bioinformatics", color=journal_colors[3])
rects5 = plt.bar(index + (bar_width*4), y_stack[4,:], bar_width, label="PLoS Computational Biology", color=journal_colors[4])
plt.xticks(index + (bar_width*2), years)
plt.xticks(rotation=45)
plt.ylabel('Number of articles')
plt.xlabel('Year')
plt.title('Evolution of number of published articles per journal and year')
plt.legend(loc=2)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

plt.tight_layout()

plt.show()

# Total citations per journal and year
cursor.execute("SELECT SUM(citations) FROM - GROUP BY source,year ORDER BY source,year ASC")
citations_per_year_bmcbio = []
citations_per_year_bmcgen = []
citations_per_year_nar = []
citations_per_year_oxfbio = []
citations_per_year_ploscb = []

for index, row in enumerate(cursor):
    if(index < 13):
        citations_per_year_bmcbio.append(row[0])
    if(index > 12 and index < 26):
        citations_per_year_bmcgen.append(row[0])
    if(index > 25 and index < 39):
        citations_per_year_nar.append(row[0])
    if(index > 38 and index < 52):
        citations_per_year_oxfbio.append(row[0])
    if(index > 51):
        citations_per_year_ploscb.append(row[0])

bar_width = 0.15
index = np.arange(len(journals))
y_stack = np.row_stack((citations_per_year_bmcbio,citations_per_year_bmcgen,citations_per_year_nar,citations_per_year_oxfbio,citations_per_year_ploscb))
colormap = plt.cm.gist_ncar 
y_pos = np.arange(len(years))
index = np.arange(13)
fig = plt.figure(figsize=(20, 24))
ax = fig.add_subplot(111)
rects1 = plt.bar(index, y_stack[0,:], bar_width, label="BMC Bioinformatics", color=journal_colors[0])
rects2 = plt.bar(index + bar_width, y_stack[1,:], bar_width, label="BMC Genomics", color=journal_colors[1])
rects3 = plt.bar(index + (bar_width*2), y_stack[2,:], bar_width, label="Nucleic Acids Research", color=journal_colors[2])
rects4 = plt.bar(index + (bar_width*3), y_stack[3,:], bar_width, label="Oxford Bioinformatics", color=journal_colors[3])
rects5 = plt.bar(index + (bar_width*4), y_stack[4,:], bar_width, label="PLoS Computational Biology", color=journal_colors[4])
plt.xticks(index + (bar_width*2), years)
plt.xticks(rotation=45)
plt.ylabel('Total citations')
plt.xlabel('Year')
plt.title('Evolution of total citations per journal and year')
plt.legend(loc=2)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

#plt.tight_layout()

plt.show()

# Total papers per topic
topics = ["Functional genomics", "Sequence analysis", "Structure analysis", "Molecular interactions, pathways and networks", "Mapping", "Molecular genetics", "Proteomics", "RNA", "Phylogeny", "Transcriptomics", "Pharmacogenomics", "Tools", "DNA"]
papers_per_topic = []
#topic_colors = []
#for i in range(0,len(topics)):
    #topic_colors.append(plt.get_cmap("tab20").jet(i))
    #topic_colors.append(generate_new_color(topic_colors,pastel_factor = 0.0))
      
cursor.execute("SELECT B.edamCategory,COUNT(DISTINCT(title)) FROM - A JOIN - B on A.topic = B.id WHERE topic != -1 GROUP BY topic ORDER BY B.id")
for row in cursor:
    papers_per_topic.append(row[1])

y_pos = np.arange(len(topics))
bars = plt.bar(y_pos, papers_per_topic, align='center', alpha=0.5, color=topic_colors)
plt.xticks(y_pos, topics)
plt.xticks(rotation=90)
plt.ylabel('Number of articles')
plt.xlabel('Topic')
plt.title('Number of published articles per topic')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

 
plt.show()

plt.pie(papers_per_topic, labels=topics, autopct='%1.0f%%', pctdistance=0.6, labeldistance=1.1, shadow=False, colors=topic_colors)
plt.title("Percentage of articles in each topic")
plt.show()

# Evolution of articles per topic in bar plot

cursor.execute("SELECT COUNT(DISTINCT(title)) FROM - A JOIN - B on A.topic = B.id WHERE topic != -1 GROUP BY topic,year ORDER BY B.id,year")

articles_per_year_funcgen = []
articles_per_year_seqan = []
articles_per_year_strucan = []
articles_per_year_molint = []
articles_per_year_mapping = []
articles_per_year_molgen = []
articles_per_year_proteomics = []
articles_per_year_rna = []
articles_per_year_phylo = []
articles_per_year_trans = []
articles_per_year_phar = []
articles_per_year_tools = []
articles_per_year_dna = []

for index, row in enumerate(cursor):
    if(index < 13):
        articles_per_year_funcgen.append(row[0])
    if(index > 12 and index < 26):
        articles_per_year_seqan.append(row[0])
    if(index > 25 and index < 39):
        articles_per_year_strucan.append(row[0])
    if(index > 38 and index < 52):
        articles_per_year_molint.append(row[0])
    if(index > 51 and index < 65):
        articles_per_year_mapping.append(row[0])
    if(index > 64 and index < 78):
        articles_per_year_molgen.append(row[0])
    if(index > 77 and index < 91):
        articles_per_year_proteomics.append(row[0])
    if(index > 90 and index < 104):
        articles_per_year_rna.append(row[0])
    if(index > 103 and index < 117):
        articles_per_year_phylo.append(row[0])
    if(index > 116 and index < 130):
        articles_per_year_trans.append(row[0])
    if(index > 129 and index < 143):
        articles_per_year_phar.append(row[0])
    if(index > 142 and index < 156):
        articles_per_year_tools.append(row[0])
    if(index > 155):
        articles_per_year_dna.append(row[0])

y_stack = np.row_stack((articles_per_year_funcgen, articles_per_year_seqan, articles_per_year_strucan, articles_per_year_molint, articles_per_year_mapping, articles_per_year_molgen, articles_per_year_proteomics, articles_per_year_rna, articles_per_year_phylo, articles_per_year_trans, articles_per_year_phar, articles_per_year_tools, articles_per_year_dna))

df = pd.DataFrame({topics[0]:y_stack[0,:],topics[1]:y_stack[1,:],topics[2]:y_stack[2,:],topics[3]:y_stack[3,:],topics[4]:y_stack[4,:],topics[5]:y_stack[5,:],topics[6]:y_stack[6,:],topics[7]:y_stack[7,:],topics[8]:y_stack[8,:],topics[9]:y_stack[9,:],topics[10]:y_stack[10,:],topics[11]:y_stack[11,:],topics[12]:y_stack[12,:]}, index=years)
dfPlot = df.plot(kind='bar', rot=45, stacked=True, cmap=cmap_tab20, title="Number of published articles per topic and per year")
dfPlot.set_xlabel("Year")
dfPlot.set_ylabel("Number of articles")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# Total citations per topic
topics = ["Functional genomics", "Sequence analysis", "Structure analysis", "Molecular interactions, pathways and networks", "Mapping", "Molecular genetics", "Proteomics", "RNA", "Phylogeny", "Transcriptomics", "Pharmacogenomics", "Tools", "DNA"]
papers_per_topic = []
  
cursor.execute("SELECT B.id,SUM(citations) FROM - A JOIN - B on A.topic = B.id WHERE topic != -1 GROUP BY topic ORDER BY B.id")
for row in cursor:
    papers_per_topic.append(row[1])

y_pos = np.arange(len(topics))
bars = plt.bar(y_pos, papers_per_topic, align='center', alpha=0.5, color=topic_colors)
plt.xticks(y_pos, topics)
plt.xticks(rotation=90)
plt.ylabel('Number of citations')
plt.xlabel('Topic')
plt.title('Number of citations per topic')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.pie(papers_per_topic, labels=topics, autopct='%1.0f%%', pctdistance=0.6, labeldistance=1.1, shadow=False, colors=topic_colors)
plt.title("Percentage of citations per topic")
plt.show()

# Evolution of citations per topic in bar plot

cursor.execute("SELECT B.id,SUM(citations) FROM - A JOIN - B on A.topic = B.id WHERE topic != -1 GROUP BY topic,year ORDER BY B.id,year")

citations_per_year_funcgen = []
citations_per_year_seqan = []
citations_per_year_strucan = []
citations_per_year_molint = []
citations_per_year_mapping = []
citations_per_year_molgen = []
citations_per_year_proteomics = []
citations_per_year_rna = []
citations_per_year_phylo = []
citations_per_year_trans = []
citations_per_year_phar = []
citations_per_year_tools = []
citations_per_year_dna = []

for index, row in enumerate(cursor):
    if(index < 13):
        citations_per_year_funcgen.append(row[1])
    if(index > 12 and index < 26):
        citations_per_year_seqan.append(row[1])
    if(index > 25 and index < 39):
        citations_per_year_strucan.append(row[1])
    if(index > 38 and index < 52):
        citations_per_year_molint.append(row[1])
    if(index > 51 and index < 65):
        citations_per_year_mapping.append(row[1])
    if(index > 64 and index < 78):
        citations_per_year_molgen.append(row[1])
    if(index > 77 and index < 91):
        citations_per_year_proteomics.append(row[1])
    if(index > 90 and index < 104):
        citations_per_year_rna.append(row[1])
    if(index > 103 and index < 117):
        citations_per_year_phylo.append(row[1])
    if(index > 116 and index < 130):
        citations_per_year_trans.append(row[1])
    if(index > 129 and index < 143):
        citations_per_year_phar.append(row[1])
    if(index > 142 and index < 156):
        citations_per_year_tools.append(row[1])
    if(index > 155):
        citations_per_year_dna.append(row[1])

y_stack = np.row_stack((citations_per_year_funcgen, citations_per_year_seqan, citations_per_year_strucan, citations_per_year_molint, citations_per_year_mapping, citations_per_year_molgen, citations_per_year_proteomics, citations_per_year_rna, citations_per_year_phylo, citations_per_year_trans, citations_per_year_phar, citations_per_year_tools, citations_per_year_dna))

df = pd.DataFrame({topics[0]:y_stack[0,:],topics[1]:y_stack[1,:],topics[2]:y_stack[2,:],topics[3]:y_stack[3,:],topics[4]:y_stack[4,:],topics[5]:y_stack[5,:],topics[6]:y_stack[6,:],topics[7]:y_stack[7,:],topics[8]:y_stack[8,:],topics[9]:y_stack[9,:],topics[10]:y_stack[10,:],topics[11]:y_stack[11,:],topics[12]:y_stack[12,:]}, index=years)
df=df.astype(float)
dfPlot = df.plot(kind='bar', rot=45, stacked=True, cmap=cmap_tab20, title="Number of citations per topic and per year")
#plt.legend(loc='upper right')
dfPlot.set_xlabel("Year")
dfPlot.set_ylabel("Number of citations")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# Number of articles per topic and journal

cursor.execute("SELECT COUNT(DISTINCT(title)) FROM - A JOIN - B on A.topic = B.id WHERE topic != -1 GROUP BY topic,source ORDER BY B.id,source")

articles_per_source_funcgen = []
articles_per_source_seqan = []
articles_per_source_strucan = []
articles_per_source_molint = []
articles_per_source_mapping = []
articles_per_source_molgen = []
articles_per_source_proteomics = []
articles_per_source_rna = []
articles_per_source_phylo = []
articles_per_source_trans = []
articles_per_source_phar = []
articles_per_source_tools = []
articles_per_source_dna = []

for index, row in enumerate(cursor):
    if(index < 5):
        articles_per_source_funcgen.append(row[0])
    if(index > 4 and index < 10):
        articles_per_source_seqan.append(row[0])
    if(index > 9 and index < 15):
        articles_per_source_strucan.append(row[0])
    if(index > 14 and index < 20):
        articles_per_source_molint.append(row[0])
    if(index > 19 and index < 25):
        articles_per_source_mapping.append(row[0])
    if(index > 24 and index < 30):
        articles_per_source_molgen.append(row[0])
    if(index > 29 and index < 35):
        articles_per_source_proteomics.append(row[0])
    if(index > 34 and index < 40):
        articles_per_source_rna.append(row[0])
    if(index > 39 and index < 45):
        articles_per_source_phylo.append(row[0])
    if(index > 44 and index < 50):
        articles_per_source_trans.append(row[0])
    if(index > 49 and index < 55):
        articles_per_source_phar.append(row[0])
    if(index > 54 and index < 60):
        articles_per_source_tools.append(row[0])
    if(index > 59):
        articles_per_source_dna.append(row[0])

y_stack = np.row_stack((articles_per_source_funcgen, articles_per_source_seqan, articles_per_source_strucan, articles_per_source_molint, articles_per_source_mapping, articles_per_source_molgen, articles_per_source_proteomics, articles_per_source_rna, articles_per_source_phylo, articles_per_source_trans, articles_per_source_phar, articles_per_source_tools, articles_per_source_dna))

df = pd.DataFrame({topics[0]:y_stack[0,:],topics[1]:y_stack[1,:],topics[2]:y_stack[2,:],topics[3]:y_stack[3,:],topics[4]:y_stack[4,:],topics[5]:y_stack[5,:],topics[6]:y_stack[6,:],topics[7]:y_stack[7,:],topics[8]:y_stack[8,:],topics[9]:y_stack[9,:],topics[10]:y_stack[10,:],topics[11]:y_stack[11,:],topics[12]:y_stack[12,:]}, index=journals)
df=df.astype(float)
dfPlot = df.plot(kind='bar', rot=45, stacked=True, cmap=cmap_tab20, title="Number of articles per topic and per journal")
#plt.legend(loc='upper right')
dfPlot.set_xlabel("Journal")
dfPlot.set_ylabel("Number of articles")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


# TOP 15 INSTITUTIONS

# Papers per institution
institutions = ["CNRS", "Chinese Academy of Sciences", "Cornell Univ.", "EMBL-EBI", "Harvard Univ.", "Imperial College London", "NIH","Stanford Univ.","Univ. of Cambridge","Univ. of Michigan", "Univ. of Oxford", "Univ. of Tokyo", "Univ. of Toronto", "Univ. of Washington", "Wellcome Trust Sanger Institute"]
papers_per_institution = []
#institution_colors = []
#for i in range(0,len(institutions)):
#      institution_colors.append(generate_new_color(institution_colors,pastel_factor = 0.5))
      
cursor.execute("SELECT total_papers FROM - WHERE id IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) GROUP BY name ORDER BY name")
for row in cursor:
    papers_per_institution.append(row[0])

y_pos = np.arange(len(institutions))
bars = plt.bar(y_pos, papers_per_institution, align='center', alpha=0.5, color=institution_colors)
plt.xticks(y_pos, institutions)
plt.xticks(rotation=90)
plt.ylabel('Number of articles')
plt.xlabel('Institution')
plt.title('Top15 institutions: number of articles')

for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# Papers per institution per year

cursor.execute("SELECT COUNT(DISTINCT(B.title)) FROM - A JOIN - B JOIN - C ON C.title = B.title AND B.idAffiliation = A.id WHERE A.id IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) AND B.idAffiliation IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) GROUP BY B.idAffiliation,C.year ORDER BY name,year ASC")

papers_per_year_ins1 = []
papers_per_year_ins2 = []
papers_per_year_ins3 = []
papers_per_year_ins4 = []
papers_per_year_ins5 = []
papers_per_year_ins6 = []
papers_per_year_ins7 = []
papers_per_year_ins8 = []
papers_per_year_ins9 = []
papers_per_year_ins10 = []
papers_per_year_ins11 = []
papers_per_year_ins12 = []
papers_per_year_ins13 = []
papers_per_year_ins14 = []
papers_per_year_ins15 = []

for index, row in enumerate(cursor):
    if(index < 13):
        papers_per_year_ins1.append(row[0])
    if(index > 12 and index < 26):
        papers_per_year_ins2.append(row[0])
    if(index > 25 and index < 39):
        papers_per_year_ins3.append(row[0])
    if(index > 38 and index < 52):
        papers_per_year_ins4.append(row[0])
    if(index > 51 and index < 65):
        papers_per_year_ins5.append(row[0])
    if(index > 64 and index < 78):
        papers_per_year_ins6.append(row[0])
    if(index > 77 and index < 91):
        papers_per_year_ins7.append(row[0])
    if(index > 90 and index < 104):
        papers_per_year_ins8.append(row[0])
    if(index > 103 and index < 117):
        papers_per_year_ins9.append(row[0])
    if(index > 116 and index < 130):
        papers_per_year_ins10.append(row[0])
    if(index > 129 and index < 143):
        papers_per_year_ins11.append(row[0])
    if(index > 142 and index < 156):
        papers_per_year_ins12.append(row[0])
    if(index > 155 and index < 169):
        papers_per_year_ins13.append(row[0])
    if(index > 168 and index < 182):
        papers_per_year_ins14.append(row[0])
    if(index > 181):
        papers_per_year_ins15.append(row[0])

y_stack = np.row_stack((papers_per_year_ins1, papers_per_year_ins2, papers_per_year_ins3, papers_per_year_ins4, papers_per_year_ins5, papers_per_year_ins6, papers_per_year_ins7, papers_per_year_ins8, papers_per_year_ins9, papers_per_year_ins10, papers_per_year_ins11, papers_per_year_ins12, papers_per_year_ins13, papers_per_year_ins14, papers_per_year_ins15))

df = pd.DataFrame({institutions[0]:y_stack[0,:],institutions[1]:y_stack[1,:],institutions[2]:y_stack[2,:],institutions[3]:y_stack[3,:],institutions[4]:y_stack[4,:],institutions[5]:y_stack[5,:],institutions[6]:y_stack[6,:],institutions[7]:y_stack[7,:],institutions[8]:y_stack[8,:],institutions[9]:y_stack[9,:],institutions[10]:y_stack[10,:],institutions[11]:y_stack[11,:],institutions[12]:y_stack[12,:],institutions[13]:y_stack[13,:],institutions[14]:y_stack[14,:]}, index=years)
dfPlot = df.plot(kind='bar', rot=45, stacked=True, cmap=cmap_tab20, title="Top15 institutions: number of published articles per year")
#plt.legend(loc='upper right')
dfPlot.set_xlabel("Year")
dfPlot.set_ylabel("Number of articles")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# Papers per institution per source

cursor.execute("SELECT COUNT(DISTINCT(B.title)) FROM - A JOIN - B JOIN - C ON C.title = B.title AND B.idAffiliation = A.id WHERE A.id IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) AND B.idAffiliation IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) GROUP BY B.idAffiliation,C.source ORDER BY name,source ASC")

papers_per_source_ins1 = []
papers_per_source_ins2 = []
papers_per_source_ins3 = []
papers_per_source_ins4 = []
papers_per_source_ins5 = []
papers_per_source_ins6 = []
papers_per_source_ins7 = []
papers_per_source_ins8 = []
papers_per_source_ins9 = []
papers_per_source_ins10 = []
papers_per_source_ins11 = []
papers_per_source_ins12 = []
papers_per_source_ins13 = []
papers_per_source_ins14 = []
papers_per_source_ins15 = []

for index, row in enumerate(cursor):
    if(index < 5):
        papers_per_source_ins1.append(row[0])
    if(index > 4 and index < 10):
        papers_per_source_ins2.append(row[0])
    if(index > 9 and index < 15):
        papers_per_source_ins3.append(row[0])
    if(index > 14 and index < 20):
        papers_per_source_ins4.append(row[0])
    if(index > 19 and index < 25):
        papers_per_source_ins5.append(row[0])
    if(index > 24 and index < 30):
        papers_per_source_ins6.append(row[0])
    if(index > 29 and index < 35):
        papers_per_source_ins7.append(row[0])
    if(index > 34 and index < 40):
        papers_per_source_ins8.append(row[0])
    if(index > 39 and index < 45):
        papers_per_source_ins9.append(row[0])
    if(index > 44 and index < 50):
        papers_per_source_ins10.append(row[0])
    if(index > 49 and index < 55):
        papers_per_source_ins11.append(row[0])
    if(index > 54 and index < 60):
        papers_per_source_ins12.append(row[0])
    if(index > 59 and index < 65):
        papers_per_source_ins13.append(row[0])
    if(index > 64 and index < 70):
        papers_per_source_ins14.append(row[0])
    if(index > 69):
        papers_per_source_ins15.append(row[0])

y_stack = np.row_stack((papers_per_source_ins1, papers_per_source_ins2, papers_per_source_ins3, papers_per_source_ins4, papers_per_source_ins5, papers_per_source_ins6, papers_per_source_ins7, papers_per_source_ins8, papers_per_source_ins9, papers_per_source_ins10, papers_per_source_ins11, papers_per_source_ins12, papers_per_source_ins13, papers_per_source_ins14, papers_per_source_ins15))

df = pd.DataFrame({institutions[0]:y_stack[0,:],institutions[1]:y_stack[1,:],institutions[2]:y_stack[2,:],institutions[3]:y_stack[3,:],institutions[4]:y_stack[4,:],institutions[5]:y_stack[5,:],institutions[6]:y_stack[6,:],institutions[7]:y_stack[7,:],institutions[8]:y_stack[8,:],institutions[9]:y_stack[9,:],institutions[10]:y_stack[10,:],institutions[11]:y_stack[11,:],institutions[12]:y_stack[12,:],institutions[13]:y_stack[13,:],institutions[14]:y_stack[14,:]}, index=journals)
df=df.astype(float)
dfPlot = df.plot(kind='bar', rot=45, stacked=True, cmap=cmap_tab20, title="Top 15 institutions: number of articles per journal")
#plt.legend(loc='upper right')
dfPlot.set_xlabel("Journal")
dfPlot.set_ylabel("Number of articles")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# Papers per institution per topic

cursor.execute("SELECT COUNT(DISTINCT(B.title)) FROM - A JOIN - B JOIN - C JOIN - D ON C.title = B.title AND B.idAffiliation = A.id AND C.topic = D.id WHERE C.topic != -1 AND A.id IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) AND B.idAffiliation IN (474,154,24,52,55,35,4,147,60,236,18,143,245,183,136) GROUP BY B.idAffiliation,C.topic ORDER BY name,D.edamCategory ASC")

papers_per_year_ins1 = []
papers_per_year_ins2 = []
papers_per_year_ins3 = []
papers_per_year_ins4 = []
papers_per_year_ins5 = []
papers_per_year_ins6 = []
papers_per_year_ins7 = []
papers_per_year_ins8 = []
papers_per_year_ins9 = []
papers_per_year_ins10 = []
papers_per_year_ins11 = []
papers_per_year_ins12 = []
papers_per_year_ins13 = []
papers_per_year_ins14 = []
papers_per_year_ins15 = []

for index, row in enumerate(cursor):
    if(index < 13):
        papers_per_year_ins1.append(row[0])
    if(index > 12 and index < 26):
        papers_per_year_ins2.append(row[0])
    if(index > 25 and index < 39):
        papers_per_year_ins3.append(row[0])
    if(index > 38 and index < 52):
        papers_per_year_ins4.append(row[0])
    if(index > 51 and index < 65):
        papers_per_year_ins5.append(row[0])
    if(index > 64 and index < 78):
        papers_per_year_ins6.append(row[0])
    if(index > 77 and index < 91):
        papers_per_year_ins7.append(row[0])
    if(index > 90 and index < 104):
        papers_per_year_ins8.append(row[0])
    if(index > 103 and index < 117):
        papers_per_year_ins9.append(row[0])
    if(index > 116 and index < 130):
        papers_per_year_ins10.append(row[0])
    if(index > 129 and index < 143):
        papers_per_year_ins11.append(row[0])
    if(index > 142 and index < 156):
        papers_per_year_ins12.append(row[0])
    if(index > 155 and index < 169):
        papers_per_year_ins13.append(row[0])
    if(index > 168 and index < 182):
        papers_per_year_ins14.append(row[0])
    if(index > 181):
        papers_per_year_ins15.append(row[0])

y_stack = np.row_stack((papers_per_year_ins1, papers_per_year_ins2, papers_per_year_ins3, papers_per_year_ins4, papers_per_year_ins5, papers_per_year_ins6, papers_per_year_ins7, papers_per_year_ins8, papers_per_year_ins9, papers_per_year_ins10, papers_per_year_ins11, papers_per_year_ins12, papers_per_year_ins13, papers_per_year_ins14, papers_per_year_ins15))

df = pd.DataFrame({institutions[0]:y_stack[0,:],institutions[1]:y_stack[1,:],institutions[2]:y_stack[2,:],institutions[3]:y_stack[3,:],institutions[4]:y_stack[4,:],institutions[5]:y_stack[5,:],institutions[6]:y_stack[6,:],institutions[7]:y_stack[7,:],institutions[8]:y_stack[8,:],institutions[9]:y_stack[9,:],institutions[10]:y_stack[10,:],institutions[11]:y_stack[11,:],institutions[12]:y_stack[12,:],institutions[13]:y_stack[13,:],institutions[14]:y_stack[14,:]}, index=topics)
dfPlot = df.plot(kind='bar', rot=90, stacked=True, cmap=cmap_tab20, title="Top15 institutions: number of articles per topic")
#plt.legend(loc='upper right')
dfPlot.set_xlabel("Topic")
dfPlot.set_ylabel("Number of articles")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

'''
# Serialize
import _pickle as cPickle
cPickle.dump(journal_colors, open("journal_colors.pickle", "wb"))
cPickle.dump(topic_colors, open("topic_colors.pickle", "wb"))
cPickle.dump(institution_colors, open("institution_colors.pickle", "wb"))
cPickle.dump(plt.get_cmap("tab20"), open("cmap_tab20.pickle", "wb"))
'''