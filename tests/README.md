# tests

pytest checks for `app/pipeline.py`: a clean photo stays clean, known damage is found, repair does no harm, color input skips colorization, photos without faces skip GFPGAN, large images run without running out of memory, and a fixed seed gives identical output. The same checks also run as labelled cells in the notebooks.
