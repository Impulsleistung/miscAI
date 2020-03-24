# git tutorial
# which file has changed
git status

# parent working directory
pwd

# Working Directory -> Staging Area -> .git Directory
# show what's been modified
git diff

diff --git a/report.txt b/report.txt
index e713b17..4c0742a 100644
--- a/report.txt
+++ b/report.txt
@@ -1,4 +1,5 @@
-# Seasonal Dental Surgeries 2017-18
+# Seasonal Dental Surgeries (2017) 2017-18
+# TODO: write new summary

# This shows:

# The command used to produce the output (in this case, diff --git). In it, a and b are placeholders meaning "the first version" and "the second version".
# An index line showing keys into Git's internal database of changes. We will explore these in the next chapter.
# --- a/report.txt and +++ b/report.txt, wherein lines being removed are prefixed with - and lines being added are prefixed with +.
# A line starting with @@ that tells where the changes are being made. The pairs of numbers are start line and number of lines (in that section of the file where changes occurred). This diff output indicates changes starting at line 1, with 5 lines where there were once 4.
# A line-by-line listing of the changes with - showing deletions and + showing additions (we have also configured Git to show deletions in red and additions in green). Lines that haven't changed are sometimes shown before and after the ones that have in order to give context; when they appear, they don't have either + or - in front of them.

# add files to staging area
git add <filename>

# compare files in staging area
git diff -r HEAD
git diff -r HEAD <path/file>

# when a file is added to the staging area, the changes shall be saved
git commit -m "SOME MESSAGE"

# show history of commits
git log
git log <file>

# Better documentation of commits WITHOUT -m "TEXT" will open an editor
git commit

# view the details of a specific commit
git show
git show 0da2e #first few characters of hash

# view details of the second recend commit
git show HEAD~1

# remove unwanted files. Warning: Unwanted = files not tracked
git status    # lists files not added
git clean -n   # lists files that would be removed
git clean -f   # finally removes files listed previously

# Change GIT configuration
# Let's get the config first
git config --list --global
	user.email=repl@datacamp.com
	user.name=Rep Loop
	core.editor=nano
# Then set new K,V pairs
git config --global user.email rep.loop@datacamp.com

# Selective staging
git add ./data/northern.csv    # We only want this one
git commit -m "Adding data from northern region."    # The general commit will do
	[master 42efe70] Adding data from northern region.
	 1 file changed, 1 insertion(+)    # Only one affected
	 
# undo changes which are not stages yet
# Changes not staged for commit:
  # (use "git add <file>..." to update what will be committed)
  # (use "git checkout -- <file>..." to discard changes in working directory)

        # modified:   data/northern.csv

git checkout -- ./data/northern.csv


# Undo changes to a stages file with the reset-checkout combination
git reset ./data/northern.csv
# Unstaged changes after reset:
# M       data/northern.csv
git checkout -- ./data/northern.csv

# Restore older version of file
# first, get the version history of the file with the hashes
git log -2 ./data/western.csv
	commit 81f8995a9fbb0cc9fddc295526a40639371269f8
	Author: Rep Loop <repl@datacamp.com>
	Date:   Wed Feb 5 16:37:55 2020 +0000

		Adding fresh data for western region.

	commit 5be6e52827e7d9b398c3701486342aa3f7ae4b58
	Author: Rep Loop <repl@datacamp.com>
	Date:   Wed Feb 5 16:37:55 2020 +0000

    Adding fresh data for southern and western regions.

# second, checkout the version that is wanted to be the latest
git checkout 5be6 data/western.csv
# third, make sure that this is the correct version of the file
cat data/western.csv
# fourth, commit the file from stage to repository
git commit data/western.csv -m "Using the older version"
	[master efb22d3] Using the older version
	 1 file changed, 3 deletions(-)

