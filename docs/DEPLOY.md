# Deploying the demo on Hugging Face Spaces

GitHub Pages cannot host this app: it only serves static files, and the app is a Python
server (Gradio, SQLite checkpoints, background runs). A Space runs the Python app for free.

## What protects the model budget

- `DEMO_PASSWORD` is a secret of the Space. In live mode the app asks for it before any call
  that spends budget: loading the proposed questions and starting a measurement.
- Five wrong guesses in one minute lock the gate for everyone until the minute passes, so the
  password cannot be brute-forced.
- The simulated demo needs no password because it calls no model.
- `SPEND_CAP_USD` limits **each measurement**, not the total. Keep it small and rotate the
  provider key if the password leaks.
- Never write the password or any key in the repository (it is public). Set them only as
  Space secrets.

## Steps

1. Create a free account at <https://huggingface.co/join> and a write token at
   <https://huggingface.co/settings/tokens> (type *Write*). Git asks for it as the password.
2. Create a Space at <https://huggingface.co/new-space>: SDK **Gradio**, hardware **CPU basic**,
   visibility as you prefer. Note its name, `USER/SPACE`.
3. In the Space, open *Settings → Variables and secrets* and add:

   | Kind | Name | Value |
   | --- | --- | --- |
   | Secret | `DEMO_PASSWORD` | the shared password |
   | Secret | `LLM_API_KEY` | your provider key |
   | Variable | `LLM_MODE` | `live` (leave unset or `simulated` for a free demo) |
   | Variable | `LLM_PROVIDER` | `groq` |
   | Variable | `SPEND_CAP_USD` | `0.50` |
   | Variable | `SEARCH_PROVIDER` | `duckduckgo` (or `none` for fixtures) |

4. A Space needs this header at the top of its `README.md`. Add it on a throwaway branch so
   the project README stays clean:

   ```yaml
   ---
   title: AI Share of Voice
   sdk: gradio
   sdk_version: 6.30.0
   python_version: "3.12"
   app_file: app.py
   ---
   ```

   ```bash
   git switch -c deploy/hf-space
   # paste the header above at the very top of README.md, then:
   git commit -am "chore(deploy): add the space header"
   git remote add hf https://huggingface.co/spaces/USER/SPACE
   git push hf deploy/hf-space:main --force
   git switch -
   ```

5. The Space builds from `requirements.txt` and starts `app.py`. The app reads
   `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT`, which the Space sets, and listens there.

## Limits to know

- The disk of a free Space is not persistent: `runs/` and the history are lost on restart.
- A free Space sleeps after a while without visits and wakes on the next one.
- Search-service quotas and the provider's own limits are separate from `SPEND_CAP_USD`.
