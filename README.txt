README!!!

###############################
Drexel University
CS-615: HW2
John Obuch
###############################

Submission File Contents:

The /HW1_Submision folder contains the following:

1) README.txt - overview of file contents and how to execute the code (i.e. THE CURRENT FILE YOU ARE READING!!!).
2) \yalefaces - Folder contents contain the provided GIF images
3) CS615_HW2.py - Mother file that produces outputs for all parts of the assignment. 
4) HW2_revised.pdf - The original outline of the homework assignment.
5) CS615_HW2.pdf - Assignement submission write-up.


To Run The Code:

The following outlines how to run the source code to reproduce the results for each
part of the assignment.

Part 1:

See PDF write-up for mathematical approach to the theory questions.

PYTHON - Part 2 - 3:

To run the CS615_HW2.py source code file on Tux, upload the the \HW2_Submssion.zip file to Tux. 
To do this, open up the command prompt terminal on your local machine and type the following command:

scp HW2_Submission.zip user_id@tux.cs.drexel.edu:~/Directory_Name

Where the user_id is your drexel uid and /Directory_Name is the directory on Tux that you will be uploading the zip file to.
Next, navigate to the Tux terminal from the command prompt via the following command:

ssh user_id@tux.cs.drexel.edu

You will be prompted to enter your Tux password credentials. 
Once in the tux envrionment, cd (change directory) into the directory where you uploaded the HW2_Submission.zip file to via the following command:

cd Directory_Name

Once in the directory of interest, type the following command to unzip the file contents:

unzip HW2_Submission.zip

Once the file has been unzipped, navigate into the /HW2_Submission via the following command:

cd HW2_Submission

Finally, once in the directory of interest (i.e. the /HW2_Submission directory), in the Tux terminal type the following command:

python3 CS615_HW2.py

Part 2: Will run automatically.
Part 3: User will be prompted to provide input for network configuration.

The results for all parts (see CS615_HW2.pdf file contents for part 1) of the assignment will populate within the terminal. 
Note: All resulting figures will be stored in the parent directory after the script has been exicuted, namely the /HW2_Submission directory.
To ensure the resulting figures populated in the parent directory, in the terminal type the following:

ls

The images (i.e. the .png files) should now appear in  the parent directory after the code has been executed if they were not already present in the parent directory.
Otherwise, if the parent directory already contained the .png images, they will be updated/overwritten.


ADDITIONAL NOTES/OBSEVATIONS/LEARNINGS:

Random seed was set at 1.
Hyperparmeters and termination threshold(s) are very sensitive and play a big factor in the accuracy of the results.
Cross OS platform (e.g. Local VS. Tux) differences in results caused by randomization/seeding and package version differences.

END OF DOCUMENT.