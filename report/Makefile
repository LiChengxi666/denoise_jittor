LATEXMK := latexmk
XELATEX_FLAGS := -xelatex -interaction=nonstopmode -halt-on-error -file-line-error
METRICS_CSV ?= ../starter_code/logs/vm_strong/metrics.csv
FIGURES := figures/system_overview.pdf figures/model_architecture.pdf \
           figures/inference_pipeline.pdf figures/cloud_experiment.pdf \
           figures/training_curve.pdf
OPTIONAL_FIGURES :=

.PHONY: all figures optional clean distclean sync-metrics

all: figures main.pdf

figures: $(FIGURES)

optional: $(OPTIONAL_FIGURES)

sync-metrics:
	@echo "Pull cloud training logs into starter_code/logs/vm_strong/ before optional figures."
	@echo "Example:"
	@echo "  rsync -avP ubuntu@36.103.236.211:~/denoise_jittor/starter_code/logs/vm_strong/ ../starter_code/logs/vm_strong/"

figures/%.pdf: figures/%.tex
	cd figures && $(LATEXMK) $(XELATEX_FLAGS) $(notdir $<)

figures/training_curve.pdf: scripts/plot_training_curve.py ../starter_code/logs/vm_strong/metrics.csv data/checkpoint_val_loss_known.csv
	python scripts/plot_training_curve.py \
	  --metrics ../starter_code/logs/vm_strong/metrics.csv \
	  --train-log ../starter_code/logs/vm_strong/train.log \
	  --checkpoints data/checkpoint_val_loss_known.csv \
	  --output figures/training_curve.pdf \
	  --total-epochs 80

main.pdf: main.tex references.bib $(FIGURES)
	$(LATEXMK) $(XELATEX_FLAGS) main.tex

clean:
	$(LATEXMK) -c main.tex
	cd figures && $(LATEXMK) -c system_overview.tex && $(LATEXMK) -c model_architecture.tex \
		&& $(LATEXMK) -c inference_pipeline.tex && $(LATEXMK) -c cloud_experiment.tex
	rm -f main.bbl

distclean: clean
	rm -f main.pdf $(FIGURES) $(OPTIONAL_FIGURES)
