"""CLI for managing TensorBoard servers in Neuracore ML framework."""

from neuracore.ml.logging.tensorboard_server_manager import (
    cleanup_tensorboard_servers,
    list_tensorboard_servers,
    start_tensorboard,
    stop_tensorboard,
)


def launch_tensorboard() -> None:
    """CLI entry point for TensorBoard management."""
    import argparse

    parser = argparse.ArgumentParser(description="Neuracore TensorBoard Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start TensorBoard server")
    start_parser.add_argument("log_dir", help="Directory containing logs")
    start_parser.add_argument("--port", type=int, default=6006, help="Port to serve on")
    start_parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser"
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop TensorBoard server")
    stop_parser.add_argument("--port", type=int, default=6006, help="Port to stop")

    # List command
    subparsers.add_parser("list", help="List running servers")

    # Stop all command
    subparsers.add_parser("stop-all", help="Stop all servers")

    args = parser.parse_args()

    if args.command == "start":
        server = start_tensorboard(
            args.log_dir, port=args.port, open_browser=not args.no_browser
        )
        print(f"TensorBoard started at {server.get_url()}")

    elif args.command == "stop":
        stop_tensorboard(args.port)
        print(f"Stopped TensorBoard on port {args.port}")

    elif args.command == "list":
        servers = list_tensorboard_servers()
        if servers:
            print("Running TensorBoard servers:")
            for port, info in servers.items():
                status = "✓" if info["running"] else "✗"
                print(f"  {status} Port {port}: {info['url']} -> {info['log_dir']}")
        else:
            print("No TensorBoard servers running")

    elif args.command == "stop-all":
        cleanup_tensorboard_servers()
        print("Stopped all TensorBoard servers")

    else:
        parser.print_help()


if __name__ == "__main__":
    launch_tensorboard()
