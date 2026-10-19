"""
Punto de entrada único:
    python -m signa check                     diagnóstico (librerías, modelos, golden, cámara)
    python -m signa collect --signer jose     grabar dataset
    python -m signa train                     entrenar, evaluar y exportar
    python -m signa run [--listen] [--round]  app del dispositivo
    python -m signa listen                    probar solo voz -> texto
    python -m signa say "hola"                probar solo texto -> voz
Opciones comunes: --config config.json --source 0|picam --dominant Right|Left --pose
"""
import argparse
import logging
import sys

from .config import DEFAULT_LABELS, Config


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.json")
    common.add_argument("--source", help="índice de webcam o 'picam'")
    common.add_argument("--dominant", choices=["Right", "Left"], help="mano dominante del señante")
    common.add_argument("--pose", action="store_true", help="registrar ubicación respecto al cuerpo")
    common.add_argument("-v", "--verbose", action="store_true")

    ap = argparse.ArgumentParser(prog="signa", description="SIGNA - deletreo LSM offline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", parents=[common]); c.add_argument("--no-camera", action="store_true")
    c = sub.add_parser("collect", parents=[common])
    c.add_argument("--signer", required=True, help="id del señante (separa entrenamiento y prueba)")
    c.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    c = sub.add_parser("train", parents=[common])
    c.add_argument("--use-location", action="store_true", help="usar ubicación (requiere datos con --pose)")
    c.add_argument("--epochs", type=int, default=300)
    c.add_argument("--aug", type=int, default=3)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--skip-eval", action="store_true")
    c = sub.add_parser("run", parents=[common])
    c.add_argument("--listen", action="store_true"); c.add_argument("--mic", type=int)
    c.add_argument("--round", action="store_true", help="diseño para pantalla circular")
    c.add_argument("--fullscreen", action="store_true")
    c = sub.add_parser("listen", parents=[common]); c.add_argument("--mic", type=int)
    c = sub.add_parser("say", parents=[common]); c.add_argument("text")
    return ap


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)
    if args.source:
        cfg.camera_source = args.source
    if args.dominant:
        cfg.dominant_hand = args.dominant
    if args.pose:
        cfg.use_pose = True

    if args.cmd == "check":
        from . import check
        return 0 if check.run(cfg, camera=not args.no_camera) else 1
    if args.cmd == "collect":
        from . import collect
        collect.run(cfg, args.signer, [l.strip() for l in args.labels.split(",") if l.strip()])
    elif args.cmd == "train":
        from . import train
        train.run(cfg, args.use_location, args.epochs, args.aug, args.seed, args.skip_eval)
    elif args.cmd == "run":
        from . import app
        app.run(cfg, args.listen, args.mic, args.round, args.fullscreen)
    elif args.cmd == "listen":
        from .stt import VoiceListener
        print("Escuchando... Ctrl+C para salir")
        try:
            VoiceListener(cfg.vosk_model, args.mic).run(on_final=lambda t: print("OYENTE:", t))
        except KeyboardInterrupt:
            pass
    elif args.cmd == "say":
        from .tts import Speaker
        sp = Speaker(cfg.piper_voice)
        sp.say(args.text)
        sp.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
