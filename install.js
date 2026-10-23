module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "FastVideo MLX runs on Apple Silicon Macs only."
      },
      next: null
    },
    {
      when: "{{!exists('FastVideo')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/hao-ai-lab/FastVideo.git FastVideo"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "FastVideo",
        message: [
          "git fetch --all --tags",
          "git checkout fa2d1324c60050cd65209c6aaeac470e7fb412f5"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -e \"./FastVideo[mlx]\"",
          "uv pip install -r app/requirements.txt"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Install complete. Click 'Start' to launch FastVideo MLX. Models download on first use."
      }
    }
  ]
}
