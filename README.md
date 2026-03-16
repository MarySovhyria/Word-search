**Word Search Puzzle**

**General Idea**
This project builds a system that can automatically solve Word Search puzzles from photos of puzzle books.

The input to the system is:

A PNG image of a Word Search puzzle.
A list of words that need to be found in the puzzle.
The output of the system is:

The position of each word in the grid, given as the row and column of the first letter and the row and column of the last letter.
To solve the puzzle, the system works in two main stages.

First, it recognises the letters in the grid.
Each puzzle image is already divided into small 20×20 pixel squares, where each square contains one letter. The system converts each square into a small feature vector (no more than 20 values). Then a trained classifier predicts which letter (A–Z) is shown in each square.

Second, once all letters are predicted, the system searches for each word in the grid. Words can appear:

Left to right
Right to left
Top to bottom
Bottom to top
Diagonally (in all diagonal directions)
The system checks all possible directions and returns the coordinates of where each word is found.

The system is designed to work on both high-quality and low-quality images.

**Files in the Project**
The project contains four Python files:

train.py
Trains the models using the training data.

evaluate.py
Tests the system on development data and reports results.

system.py
Contains the main logic of the system:

Dimensionality reduction
Letter classification
Word search algorithm
This is the only file that was modified.
utils.py
Contains helper functions for loading and splitting images. This file is not changed.

When training is complete, two model files are saved in the data/ folder:

model.high.json.gz
model.low.json.gz
These are used during evaluation.

**How to Run the Code**

Open a terminal and navigate to the folder containing train.py and evaluate.py.
  
 **Train the models**
 
  cd path/to/assignment/code
  python train.py
  
  **Evaluate the system**
  
  python evaluate.py
  
  **Evaluate and display solved puzzles**
  
  python evaluate.py --display
