% Plantilla para la memoria del TFG
% Plantilla desarrollada por Juan Gutiérrez Aguado (www.uv.es/jgutierr)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tfg}[2020/03/26 Plantilla para la memoria del TFG]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass{book}

\newcommand{\@tutorlabel}{\ClassError{tfg}{Falta especificar
\string\tutorlabel{}}
 {Usa el comando \string\tutorlabel{} en el preambulo para especificar
\MessageBreak la etiqueta asociada (Tutor, Tutora, Tutores)}}
\newcommand{\tutorlabel}[1]{\gdef\@tutorlabel{#1}}

\newcommand{\@authorlabel}{\ClassError{tfg}{Falta especificar
\string\autorlabel{}}
 {Usa el comando \string\authorlabel{} en el preambulo para especificar
\MessageBreak la etiqueta asociada (Autor, Autora)}}
\newcommand{\authorlabel}[1]{\gdef\@authorlabel{#1}}

\newcommand{\@tutor}{\ClassError{tfg}{Falta especificar
\string\tutor{}}
 {Usa el comando \string\tutor{} en el preambulo para especificar
\MessageBreak el nombre del tutor o la tutora.}}
\newcommand{\tutor}[1]{\gdef\@tutor{#1}}

\newcommand{\@convocatoria}{\ClassError{tfg}{Falta especificar
\string\convocatoria{}}
 {Usa el comando \string\convocatoria{} en el preambulo para especificar
\MessageBreak el mes y año de la lectura.}}
\newcommand{\convocatoria}[1]{\gdef\@convocatoria{#1}}


\newcommand{\@titulacion}{\ClassError{tfg}{Falta especificar
\string\titulación{}}
 {Usa el comando \string\titulacion{} en el preambulo para especificar
\MessageBreak la titulacion.}}
\newcommand{\titulacion}[1]{\gdef\@titulacion{#1}}


\RequirePackage[T1]{fontenc}
\RequirePackage[spanish,activeacute,es-tabla]{babel}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{fancyhdr}
\RequirePackage{listings}
\RequirePackage{hyperref}
\RequirePackage[pdftex]{graphicx}
\RequirePackage{tikz}
\RequirePackage{framed}
\RequirePackage{geometry}
\RequirePackage{soul}
\RequirePackage{bold-extra}
\RequirePackage{ragged2e}
\RequirePackage{longtable}


\usepackage[utf8]{inputenc} %Codificacion de entrada

% Ruta donde buscar las figuras
\graphicspath{{figs/}}
\DeclareGraphicsExtensions{.pdf,.jpg,.png}


 \geometry{
  a4paper,
  total={160mm,255mm},
  left=25mm,
  right=25mm,
  top=25mm,
  bottom=20mm
 }

\newlength{\ancho}
\setlength{\ancho}{16cm}


\hypersetup{bookmarks=true,%
bookmarksnumbered=true,%
colorlinks,%
citecolor=red,%
filecolor=blue,%
linkcolor=blue,%
urlcolor=blue,%
pdfstartview=FitH,%
pdffitwindow=true,%
pdfauthor=\@author}

\setlength{\parskip}{5pt}



% ============================================================================
% CONFIGURACIÓN DE COLORES
% ============================================================================
\definecolor{uv-blue}{RGB}{54, 95, 145}
\definecolor{uv-dark}{RGB}{36, 63, 96}
\definecolor{uv-light}{RGB}{211, 223, 238}
\definecolor{accent}{RGB}{192, 80, 77}

% ============================================================================
% DEFINICIÓN DE COMANDOS PERSONALIZADOS
% ============================================================================
\newcommand{\titulo}[1]{\textbf{\Large\color{uv-dark}#1}}
\newcommand{\subtitulo}[1]{\textbf{\small\color{uv-blue}#1}}
\newcommand{\seccion}[1]{\large\textbf{\color{uv-dark}#1}}
\newcommand{\campo}[1]{\underline{#1}}

% Para checkmark
% \usepackage{dingbat}

\newcommand{\portada}{
  \pagestyle{empty}
   \makeatletter
    \centering % Centre everything on the title page
     % Use small caps for all text on the title page

   \begin{center}
     \includegraphics[width=0.55\ancho]{figs/etse_titulo.png}
   \end{center}

   \vspace*{1\baselineskip} % White space

   \begin{minipage}[h]{\ancho}
     \centering
     \LARGE
     \capsdef{T1/ppl/m/n/5-15}{\scshape}{.16em}{.6em}{.2em}
     \scshape \@titulacion
   \end{minipage}


   \vspace*{2\baselineskip} % White space

   \begin{center}
      \includegraphics[width=0.4\ancho]{figs/escudo_logo_UV_narrow.png}
   \end{center}

   \vspace*{3\baselineskip} % White space

    {\scshape \LARGE Trabajo Fin de Grado}

   \vspace*{2\baselineskip} % White space

    % ------------------------------------------------
    %   Title
    %------------------------------------------------
    \rule{\ancho}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt} % Thick horizontal rule
    \rule{\ancho}{0.4pt} % Thin horizontal rule

    \vspace{0.75\baselineskip} % Whitespace above the title

   \begin{minipage}[h]{\ancho}
     \centering
     \LARGE
     \capsdef{T1/ppl/m/n/5-15}{\scshape}{.16em}{.6em}{.2em}
     \scshape \@title
   \end{minipage}

    \vspace{0.75\baselineskip} % Whitespace below the title

    \rule{\ancho}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt} % Thin horizontal rule
    \rule{\ancho}{1.6pt} % Thick horizontal rule

   \vfill

   \hspace*{0.15\ancho}\begin{minipage}[h]{0.85\ancho}
   \large
    \bfseries{\hspace{5cm}\scshape \@authorlabel:\\ % Author
    \mbox{}\hspace{5cm}\@author}

    \vspace{0.5\baselineskip} % Whitespace below the editor list

    {\hspace{5cm}\scshape \@tutorlabel:\\ % Tutor
    \mbox{}\hspace{5cm}\@tutor}

    % \vspace{0.5\baselineskip} % Whitespace below the editor list

    % {\scshape \@convocatoria}
 \end{minipage}
 \makeatother
 }


 \newcommand{\contraportada}{
   \pagestyle{empty}
   \makeatletter
    \centering % Centre everything on the title page
    \vspace*{0.2\baselineskip} % White space at the top of the page

   \begin{center}
     \includegraphics[width=\ancho]{figs/logo-contraportada.png}
   \end{center}

   \vspace*{\baselineskip} % White space

   \titulacion

   \vspace*{2\baselineskip} % White space

    {\scshape \LARGE Trabajo Fin de Grado}

   \vspace*{2\baselineskip} % White space

    % ------------------------------------------------
    %   Title
    %------------------------------------------------
    \rule{\ancho}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt} % Thick horizontal rule
    \rule{\ancho}{0.4pt} % Thin horizontal rule

    \vspace{2\baselineskip} % Whitespace above the title

   \begin{minipage}[h]{\ancho}
     \centering
     \LARGE
     \capsdef{T1/ppl/m/n/5-15}{\scshape}{.16em}{.6em}{.2em}
     \scshape \@title
   \end{minipage}

    \vspace{0.5\baselineskip} % Whitespace below the title

    \rule{\ancho}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt} % Thin horizontal rule
    \rule{\ancho}{1.6pt} % Thick horizontal rule

    \vspace{5\baselineskip} % Whitespace after the title block

   \hspace*{0.15\ancho}\begin{minipage}[h]{0.85\ancho}
   \large
    \bfseries{\scshape \@authorlabel: \@author} % Author

    \vspace{0.5\baselineskip} % Whitespace below the editor list

    {\scshape \@tutorlabel: \@tutor} % Tutor

    \vspace{0.5\baselineskip} % Whitespace below the editor list
 \end{minipage}

    \vfill

    \rule{\ancho}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt}

    \vspace{2\baselineskip} % Whitespace after the title block

%   {\bf TRIBUNAL}

%   \vspace{0.3\baselineskip} % Whitespace after the title block

%   \flushleft PRESIDENTE/A: \hspace*{0.5\ancho}  VOCAL 1:

%   \vspace{5\baselineskip}

%   \flushleft VOCAL 2:      \hspace*{0.6\ancho}\begin{minipage}[t]{0.3\ancho}FECHA DE DEFENSA: \\[1cm]
%                                                      CALIFICACIÓN:
%                              \end{minipage}

   \vfill
   \makeatother
   \mdseries
 }


\newcommand{\ia}{\input{tex/declaracion_ia.tex}}


 \newenvironment{resumen}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Resumen:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }

 \newenvironment{abstract}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Abstract:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }


 \newenvironment{resum}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Resum:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }


  \newenvironment{agradecimientos}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Agradecimientos:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }


\lstdefinestyle{nonumbers} {numbers=none}
\lstset{basicstyle=\scriptsize\ttfamily}
\renewcommand{\UrlFont}{\scriptsize}

% Formato de los listados de código
\lstset{
  basicstyle=\ttfamily\small,
  columns=flexible,
  keywordstyle=\color{blue}\bf,
  commentstyle=\color{gray},
  stringstyle=\color{green},
  breaklines=true,inputencoding=utf8,
  frame=single,showstringspaces=false,
  style=nonumbers,extendedchars=true,
  literate={á}{{\'a}}1 {é}{{\'e}}1 {í}{{\'i}}1 {ó}{{\'o}}1 {ú}{{\'u}}1 {Á}{{\'A}}1 {É}{{\'E}}1 {Í}{{\'I}}1 {Ó}{{\'O}}1 {Ú}{{\'U}}1 {ñ}{{\~n}}1 {Ñ}{{\~N}}1,
}

\definecolor{enlaces}{rgb}{0.6,0.6,0.9}
\definecolor{colurl}{rgb}{0.34,0.36,0.60}

\hypersetup{
    colorlinks = true,
    linkcolor = enlaces,
    citecolor = enlaces,
    filecolor = enlaces,
    urlcolor = enlaces
}

\renewcommand{\lstlistingname}{Listado}
\renewcommand{\lstlistlistingname}{Indice de listados de código}

\lstloadlanguages{html,java,sql,c,xml,tex,matlab}
\lstdefinestyle{numbers} {numbers=left, stepnumber=1, numberstyle=\tiny, numbersep=10pt}
\lstdefinestyle{nonumbers} {numbers=none}


%% Para acentos en listings con UTF8
\lstset{
    inputencoding=utf8,
    extendedchars=true,
    literate={á}{{\'a}}1 {é}{{\'e}}1 {í}{{\'i}}1 {ó}{{\'o}}1 {ú}{{\'u}}1 {Á}{{\'A}}1 {É}{{\'E}}1 {Í}{{\'I}}1 {Ó}{{\'O}}1 {Ú}{{\'U}}1 {ñ}{{\~n}}1 {Ñ}{{\~N}}1
}


%%%%%%%
%%%%%%%   DEFINICION DE UN CODIGO SIN KEYWORDS
%%%%%%%
\lstdefinelanguage{plain}
   {morekeywords={},
   sensitive=false,
}

\pagestyle{fancy}

\fancypagestyle{tfg}{%
  %Información del capítulo en la parte exterior
  \fancyhead[EL,OR]{Capítulo \thechapter}
  % Información del número de página en la parte interior
  \fancyhead[OL,ER]{Página \thepage}
  \renewcommand{\headrulewidth}{1pt}
  \fancyfoot{}
}

\fancypagestyle{appendix}{%
  %Información del capítulo en la parte exterior
  \fancyhead[EL,OR]{Apéndices}
  % Información del número de página en la parte interior
  \fancyhead[OL,ER]{Página \thepage}
  \renewcommand{\headrulewidth}{1pt}
  \fancyfoot{}
}


\fancypagestyle{plain}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
}
