# MWUMiner
Multi-Word Units Miner, based on different approaches

This repository is meant to hold code, reference data and evaluation tools for MWU detection and processing based on different approaches. Among the approaches under investigation are:
  - distance-based metrics: spread and flexibility (El Maarouf & Oakes, 2015)
  - eco-diversity inspired indices: Shannon-Wiener's Entropy and Equitability, Simpson's D1, D2 and Evenness
  - (next version) graph-based approaches: shortest path, PageRank adaptations to quantitative language processing.

As of version alpha, this is how you invocate the Multi-Word Units Miner:

USAGE: java -jar MWUMiner-version /path/to/file.txt

file.txt is a file containing the occurrences to parse

Occurrences must be delimited by a <match> ... </match> tag

MWUMiner expects one occurrence per line of input

and a tokenized and regularized input for predictable performance, eg:
```XML	 
bla bla, bla <match> my match </match> bla ( bla bla ) 
bla bla <match> my other : match </match> bla bla - bla
```

This is the usage information you get when you type java -jar MWUMiner-alpha.jar without any argument, or with either -h, --h, -help, -Help, -info, etc.

In order to replicate the results found in the "baseline" folder, you need:

  1. a text file, with one occurrence per line of the MWU under scrutiny
  2. each occurrence must be enclosed within <match> ... </match> tags
  3. for better results, the file must be tokenized, cleaned, with all words possibly reduced to lower case (tr '[:upper:] [:lower:] < text' will do the trick.
  
  If there are multiple occurrences per lines of input, each line must repeated, with exactly 1 occurrence per line, eg where the target MWU is "in fact":
  ```XML
    <match> in fact </match> , this is so obvious that people will in fact often repeat themselves
    in fact , this is so obvious that people will <match> in fact </match>  often repeat themselves    
 
