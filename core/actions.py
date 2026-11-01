"""Ações que a IA pode pedir. Regras: só o que está na lista de config.yaml, sem shell, e com confirmação."""
import shlex
import subprocess


class Actions:
    def __init__(self, cfg, confirm):
        self.c = cfg["actions"]
        self.confirm = confirm  # função(texto) -> bool (abre um diálogo para o usuário)

    def app_names(self):
        return list((self.c.get("apps") or {}).keys())

    def run(self, name, arg):
        """Retorna uma mensagem curta com o resultado (para mostrar no balão)."""
        if name != "abrir":
            return f"(ação desconhecida ignorada: {name})"
        apps = self.c.get("apps") or {}
        cmd = apps.get(arg.strip().lower())
        if not cmd:
            return f"(não tenho '{arg}' na lista de apps permitidos)"
        if self.c.get("confirm", True) and not self.confirm(f"Deixar o assistente abrir '{arg}'?"):
            return "(ação cancelada por você)"
        try:
            subprocess.Popen(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)  # sem shell=True: nada de injeção de comandos
            return f"(abrindo {arg})"
        except OSError as exc:
            return f"(não consegui abrir {arg}: {exc})"
