TC-13 now treats a changed `file_type` as a distinct retry instead of reporting that the model
repeated the same search. Its mock also honors the filter, so a PDF search cannot return the DOCX
fixture.
