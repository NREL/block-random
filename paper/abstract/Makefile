# Define the documents to be constructed and the desired file types
DOCUMENTS=abstract

SUFFIX=pdf
OUTPUTS=$(addsuffix .$(SUFFIX),$(DOCUMENTS))
all: $(OUTPUTS)

# Build rawfigs/*.eps into figs/
FIGS_EPS=$(patsubst rawfigs/%,figs/%,$(wildcard rawfigs/*.eps))
$(FIGS_EPS) : figs/%.eps : rawfigs/%.eps ; cp $< $@
FIGURES += $(FIGS_EPS)

# Build rawfigs/*.pdf into figs/
FIGS_PDF=$(patsubst rawfigs/%,figs/%,$(patsubst %.pdf,%.eps,$(wildcard rawfigs/*.pdf)))
$(FIGS_PDF) : figs/%.eps : rawfigs/%.pdf ; pdftops -eps $< $@
FIGURES += $(FIGS_PDF)

# Build rawfigs/*.png into figs/
FIGS_PNG=$(patsubst rawfigs/%,figs/%,$(patsubst %.png,%.eps,$(wildcard rawfigs/*.png)))
$(FIGS_PNG) : figs/%.eps : rawfigs/%.png ; convert $< eps3:$@
FIGURES += $(FIGS_PNG)

# Build rawfigs/*.jpg into figs/
FIGS_JPG=$(patsubst rawfigs/%,figs/%,$(patsubst %.jpg,%.eps,$(wildcard rawfigs/*.jpg)))
$(FIGS_JPG) : figs/%.eps : rawfigs/%.jpg ; convert $< eps3:$@
FIGURES += $(FIGS_JPG)

# Build rawfigs/*.ps into figs/
FIGS_PS =$(patsubst rawfigs/%,figs/%,$(patsubst %.ps, %.eps,$(wildcard rawfigs/*.ps)))
$(FIGS_PS)  : figs/%.eps : rawfigs/%.ps  ; cat $< | ps2eps --accuracy --warnings > $@
FIGURES += $(FIGS_PS)

# Build rawfigs/*.svg into figs/
FIGS_SVG=$(patsubst rawfigs/%,figs/%,$(patsubst %.svg,%.eps,$(wildcard rawfigs/*.svg)))
$(FIGS_SVG) : figs/%.eps : rawfigs/%.svg ; env --unset=DISPLAY inkscape -z $< --export-eps=$@ --export-dpi=300
FIGURES += $(FIGS_SVG)

# Ensure figs/ exists before using it, clean up figs/, and build figures before docs
$(FIGURES): | figs-directory-mkdir
clean:      | figs-directory-rm
figs-directory-mkdir: ; @mkdir -p figs
figs-directory-rm:    ; @rm -rf figs
$(addsuffix .dvi,$(DOCUMENTS)) : $(FIGURES)

# PS/PDF adapted from http://www.grassbook.org/neteler/highres_pdf.html
LATEXMK?=latexmk -f
#LATEXMK?=latexmk
DVIPS=dvips -j0 -Ppdf -u ps2pk.map -G0 -D 1200 -Z -mode ljfzzz
GSPDF=gs -q -dNOPAUSE -dBATCH -dSAFER -sOutputFile=$@ \
            -sDEVICE=pdfwrite                         \
            -dAntiAliasColorImages=false              \
            -dAntiAliasGrayImages=false               \
            -dAntiAliasMonoImages=false               \
            -dAutoFilterColorImages=false             \
            -dAutoFilterGrayImages=false              \
            -dAutoFilterMonoImages=false              \
            -dColorImageDownsampleThreshold=1.50000   \
            -dColorImageDownsampleType=/Average       \
            -dColorImageFilter=/FlateEncode           \
            -dColorImageResolution=1200               \
            -dCompatibilityLevel=1.3                  \
            -dCompressPages=true                      \
            -dDetectBlends=true                       \
            -dDownsampleColorImages=true              \
            -dDownsampleGrayImages=true               \
            -dDownsampleMonoImages=true               \
            -dEmbedAllFonts=true                      \
            -dGrayImageDownsampleThreshold=1.50000    \
            -dGrayImageDownsampleType=/Average        \
            -dGrayImageFilter=/FlateEncode            \
            -dGrayImageResolution=1200                \
            -dMonoImageDownsampleThreshold=1.50000    \
            -dMonoImageDownsampleType=/Average        \
            -dMonoImageFilter=/FlateEncode            \
            -dMonoImageResolution=1200                \
            -dOptimize=true                           \
            -dPDFSETTINGS=/prepress                   \
            -dProcessColorModel=/DeviceGray           \
            -dSubsetFonts=true                        \
            -dUseFlateCompression=false               \
            -sPAPERSIZE=letter

.PRECIOUS : %.dvi %.ps %.pdf
%.dvi     : %.tex FORCE ;    $(LATEXMK) -dvi -ps- -pdf- "-deps-out=$*.dep" $< \
                          || $(LATEXMK) -dvi -ps- -pdf-                    $<
%.ps      : %.dvi       ; $(DVIPS) $<
%.pdf     : %.ps        ; $(GSPDF) $<

# Use inotifywait to re-run 'make all' whenever any input changes
# Depends on a newer latexmk version supporting the -deps-out option
continuous: all ; bash -c "while true; do inotifywait -e close_write -e delete_self -e move --fromfile <(grep -h '^[[:blank:]]' *.dep | sed -e 's/^[[:blank:]]*//' -e 's/\\\$\//'| sort -u) || break;  $(MAKE) all; done"

clean : FORCE
	@for o in $(OUTPUTS); do $(LATEXMK) -quiet -f -C $$o & done ; wait
	@rm -f $(addsuffix .out,$(DOCUMENTS)) $(addsuffix .dep,$(DOCUMENTS))
	@rm -f *.aux *.aux.bak *.bbl *.tdo *.nav *.snm *.spl

FORCE : # http://www.gnu.org/software/make/manual/html_node/Force-Targets.html
