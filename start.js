module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "../env",
        env: {
          FASTH3_STUDIO_MODELS: "../models",
          FASTH3_STUDIO_DATA: "../data",
          FASTH3_STUDIO_FASTVIDEO: "../FastVideo",
          PYTHONUNBUFFERED: "1"
        },
        path: "app",
        message: [
          "python server.py"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
