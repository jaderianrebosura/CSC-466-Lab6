# CSC 466 Lab 6 - Collaborative Filtering

## Team Members
* Julien Fiechter
* Jaderian Rebosura

## Data

The program uses the Jester ratings dataset in .xls format, as the link to download the csv
version was not functioning properly.

jester-data-1.xls

## Executable Programs

### EvaluateCFRandom.py

Usage: python EvaluateCFRandom.py Method Size Repeats

Parameters:

* Method: Collaborative filtering method ID
* Size: Number of random test cases
* Repeats: Number of times the experiment is repeated

Example: python EvaluateCFRandom.py 2 100 5

### EvaluateCFList.py

Usage: python EvaluateCFList.py Method Filename

Parameters:

* Method: Collaborative filtering method ID
* Filename: CSV file containing test cases

Example: python EvaluateCFList.py 2 test_file.csv

Each line contains of the csv contains: UserID,ItemID

## Implemented Methods

### Method 1: Weighted Sum using Cosine Similarity

### Method 2: Adjusted Weighted N Nearest Neighbors using Pearson Correlation

### Method 3: Weighted N Nearest Neighbors using Cosine Similarity

### Method 4: Average N Nearest Neighbors Ranking

