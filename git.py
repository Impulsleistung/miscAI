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

