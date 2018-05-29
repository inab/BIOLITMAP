# BIOLITMAP: a web-based geolocated and temporal visualization of the evolution of bioinformatics publications

## DESCRIPTION

Code for the web-based geolocated and temporal visualization of bioinformatics research (BIOLITMAP): http://socialanalytics.bsc.es/biolitmap/.

Paper submitted to Oxford Bioinformatics.

## DIRECTORY STRUCTURE

* In /data the export of the BIOLITMAP SQL database is stored, as it was in December 2017.
* In /scripts the tools used for the NLP tasks are stored
** /scripts/src stores the codes related to the NLP, Clustering, Topic Modeling and Perplexity Analysis tasks.
** /scripts/vis stores the visualization created using the final Latent Dirichlet Allocation model, by employing the pyLDAvis package.

## Getting the raw source data from Scopus

In order to obtain the raw source data with which this application has been built on, the following steps need to be followed:

1) Access to the Scopus (www.scopus.com) document search tool
2) Search the documents by using the following query: 

``ISSN ( 'JOURNAL\_ISSN' ) AND ( LIMIT-TO ( PUBYEAR , 2017 ) OR LIMIT-TO ( PUBYEAR , 2016 ) OR LIMIT-TO ( PUBYEAR , 2015 ) OR LIMIT-TO ( PUBYEAR , 2014 ) OR LIMIT-TO ( PUBYEAR , 2013 ) OR LIMIT-TO ( PUBYEAR , 2012 ) OR LIMIT-TO ( PUBYEAR , 2011 ) OR LIMIT-TO ( PUBYEAR , 2010 ) OR LIMIT-TO ( PUBYEAR , 2009 ) OR LIMIT-TO ( PUBYEAR , 2008 ) OR LIMIT-TO ( PUBYEAR , 2007 ) OR LIMIT-TO ( PUBYEAR , 2006 ) OR LIMIT-TO ( PUBYEAR , 2005 ) ) AND ( LIMIT-TO ( EXACTKEYWORD , "Article" ) )``

3) Export the documents in CSV format with the 'Export' option.

## CONTACT

You can contact the developers by sending an email to adrian.bazaga@bsc.es or maria.rementeria@bsc.es

## PREVIEW

<div style="text-align:center"><img src="https://i.imgur.com/iIvs1P8.png" /></div>

