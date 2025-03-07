% ====================================================================================================
%                                Assignment Document class
% ====================================================================================================

% This template class is to be used for all homework submissions.
%
% DO NOT MODIFY THIS CLASS
%
% ------------------------------------------------------------


% ------------------------------------------------------------
% Supply Class
% ------------------------------------------------------------
\ProvidesClass{Assignment}[A class for Assignment document preparation]
% ------------------------------------------------------------


% ------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Load base class with options
% ------------------------------------------------------------
% Execute default options
\ExecuteOptions{12pt, a4paper}
% ----------
% Process given options
\ProcessOptions\relax
% ----------
% Load base with options
\LoadClass[12pt, a4paper]{article}
% ------------------------------------------------------------


% ------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Logical Structure
% ------------------------------------------------------------
\RequirePackage{ifthen} % provides \ifthenelse test
\RequirePackage{xifthen} % provides \isempty test
% ------------------------------------------------------------


% ------------------------------------------------------------
% Dates
% ------------------------------------------------------------
\RequirePackage[UKenglish,iso]{isodate}
\cleanlookdateon
% ------------------------------------------------------------


% ------------------------------------------------------------
% Document Metaparameters
% ------------------------------------------------------------
\brokenpenalty=10000
\clubpenalty=10000
\emergencystretch=\maxdimen
\hbadness=10000
\hyphenpenalty=10000
\interfootnotelinepenalty=10000 % Prevents footnotes being split onto two pages
\tolerance=1
\widowpenalty=10000
% ------------------------------------------------------------


% ------------------------------------------------------------
% Styling
% ------------------------------------------------------------
% Page
\pagestyle{plain}
% ----------
% Bibliography Style
\bibliographystyle{unsrt}
% ----------
% Fonts
\RequirePackage{concrete}
\DeclareFontSeriesDefault[rm]{bf}{sbc}
\RequirePackage[OT1]{fontenc}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Page Dimensions
% ------------------------------------------------------------
\RequirePackage{fullpage}

% Page Dimensions
\RequirePackage[
    a4paper,
    left=20mm,
    right=20mm,
    top=20mm,
    bottom=30mm,
    ]{geometry}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Tables
% ------------------------------------------------------------
\RequirePackage{tabularx}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Mathematics Macros
% ------------------------------------------------------------
\RequirePackage{amsmath}
\RequirePackage{tensor}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Display breaks
% ------------------------------------------------------------
\allowdisplaybreaks
% ------------------------------------------------------------


% ------------------------------------------------------------
% Graphics and Colours
% ------------------------------------------------------------
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{subcaption}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Document Data Capture
% ------------------------------------------------------------
\gdef\@studentname{\textcolor{red}{ENTER YOUR NAME}}
\gdef\@studentnumber{\textcolor{red}{ENTER STUDENT NUMBER}}
\gdef\@coursecode{\textcolor{red}{ENTER THE COURSE CODE}}
\gdef\@coursename{\textcolor{red}{ENTER THE COURSE NAME}}
\gdef\@assignmentnumber{\textcolor{red}{ENTER THE ASSIGNMENT NUMBER}}
\gdef\@assignmentname{\textcolor{red}{ENTER THE ASSIGNMENT NAME}}
\gdef\@duedate{\textcolor{red}{ENTER THE DUE DATE}}
% ------------------------------------------------------------


% ------------------------------------------------------------
% Document Data Reassignment
% ------------------------------------------------------------
\newcommand{\StudentName}  [1]{ \ifthenelse{ \isempty{#1} } {} { \gdef\@studentname{#1} } }
\newcommand{\StudentNumber}[1]{ \ifthenelse{ \isempty{#1} } {} { \gdef\@studentnumber{#1} } }
\newcommand{\CourseCode}[1]{ \ifthenelse{ \isempty{#1} } {} { \gdef\@coursecode{#1} } }
\newcommand{\CourseName}[1]{ \ifthenelse{ \isempty{#1} } {} { \gdef\@coursename{#1} } }
\newcommand{\AssignmentNumber}[1]{ \ifthenelse{ \isempty{#1} } {} { \gdef\@assignmentnumber{#1} } }
\newcommand{\AssignmentName}[1]  { \ifthenelse{ \isempty{#1} } {} { \gdef\@assignmentname{#1} } }
\newcommand{\DueDate}[1] { \ifthenelse{ \isempty{#1} } {} { \gdef\@duedate{#1} } }
% ------------------------------------------------------------


% ------------------------------------------------------------
% Title Page
% ------------------------------------------------------------


% ------------------------------------------------------------


% ------------------------------------------------------------
% Page Headers and Footers
% ------------------------------------------------------------
\RequirePackage{lastpage}
\RequirePackage{fancyhdr} % Custom headers and footers
% ----------
\pagestyle{fancy} % Makes all pages in the document conform to the custom headers and footers
% ----------
\fancyhead{} % No page header - if you want one, create it in the same way as the footers below
% ----------


\fancyfoot[R]{\textbf{\thepage \ of \pageref{LastPage}}} % Page numbering for right footer
% ----------
\renewcommand{\headrulewidth}{0.0pt}
\renewcommand{\footrulewidth}{0.5pt}
% ------------------------------------------------------------


% ----------
\endinput
% EOF
