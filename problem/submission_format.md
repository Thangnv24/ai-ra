# Submission Format

Submit one `output.zip` file. When unzipped it must contain:

```text
output/
  1.json
  2.json
  ...
  100.json
```

This repository creates the archive with:

```bash
python -m medkg.submit --output-dir outputs --submission-dir submission
```

The resulting file is `submission/output.zip`.
