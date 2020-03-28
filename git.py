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


# remove all files from staging area
$ git status
	On branch master
	Changes to be committed:
	  (use "git reset HEAD <file>..." to unstage)

			modified:   data/eastern.csv
			modified:   data/northern.csv
			modified:   report.txt

	$ git reset
	Unstaged changes after reset:
	M       data/eastern.csv
	M       data/northern.csv
	M       report.txt

git status
	On branch master
	Changes not staged for commit:
	  (use "git add <file>..." to update what will be committed)
	  (use "git checkout -- <file>..." to discard changes in working directory)

			modified:   data/eastern.csv
			modified:   data/northern.csv
			modified:   report.txt

	no changes added to commit (use "git add" and/or "git commit -a")
# now, the files are unstaged. To restore the lates commit, a checkout needs to be done
git checkout -- .

----------------------------------------------------

# List all branches
git branch

# View differences between branches
git diff BRANCH_A..BRANCH_B

# switch back and forth between branches
	# first, what branches are offered
git branch
	  alter-report-title
	* master
	  summary-statistics
	# second, switsch to the branch
git checkout summary-statistics
	Switched to branch 'summary-statistics'
	# third, delete a file in the branch, we just switched to
git rm report.txt
	rm 'report.txt'
	# fourth, commit the change to to branch
git commit -m "Removing report"
	[summary-statistics 91a2a36] Removing report
	 1 file changed, 7 deletions(-)
	 delete mode 100644 report.txt
	# fifth, switch back to the master branch and check if file exists there
git checkout master
	Switched to branch 'master'
ls -R
	.:
	bin  data  report.txt  results
	
# crate and switch to new branch
git checkout -b deleting-report
	Switched to a new branch 'deleting-report'
	# second, remove some file
git rm report.txt
	rm 'report.txt'
	# third, update the specific branch
git commit -m "File deleted in new branch"
	[deleting-report 29372bb] File deleted in new branch
	 1 file changed, 7 deletions(-)
	 delete mode 100644 report.txt
	# fourth, see the difference between master and the specific branch
git diff master..deleting-report
	diff --git a/report.txt b/report.txt
	deleted file mode 100644
	index 4c0742a..0000000
	--- a/report.txt
	+++ /dev/null


# merging two branches
git merge summary-statistics master # source destination
	Merge made by the 'recursive' strategy.

# resolve merge conflict 
# first, get current position
git branch
	  alter-report-title
	* master
	  summary-statistics
# second, start to merge
git merge alter-report-title master
	Auto-merging report.txt
	CONFLICT (content): Merge conflict in report.txt
	Automatic merge failed; fix conflicts and then commit the result.
# third, call status
git status
	On branch master
	You have unmerged paths.
	  (fix conflicts and run "git commit")

	Unmerged paths:
	  (use "git add <file>..." to mark resolution)

			both modified:   report.txt

	no changes added to commit (use "git add" and/or "git commit -a")
# fourth, the previous procedures have led GIT to comment directly in the concerning file. So just open and jump to the GIT-marked line. Fix by hand
nano report.txt

# fifth, the fixed file has to be staged
git add report.txt

# sixth, the staged file with the resolved conflict has to be commited
git commit -m "Conflict resolved"
	[master 1580f4b] Conflict resolved

---------------------------------------------------

# create new repository
git init REPONAME
git init # A plain call will establish one in the PWD

# clone a repository
git clone /existing/project newprojectname

# get the adress of the original repo which was cloned
git remote -v

