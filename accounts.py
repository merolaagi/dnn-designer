#!/usr/bin/env python3
"""Manage accounts from the command line.

Whoever runs the server can already read every file it stores, so there is no
security lost in letting them reset a password here — and without it, a
forgotten password would mean editing JSON by hand or losing the workspace.

    python accounts.py list
    python accounts.py add <name>
    python accounts.py passwd <name>
    python accounts.py remove <name>          keeps the workspace
    python accounts.py remove <name> --purge  deletes it too
    python accounts.py off                    removes every account

Turning accounts off leaves every workspace on disk. The shared workspace — the
one the first account adopted — becomes what the app opens again.
"""

from __future__ import annotations

import getpass
import shutil
import sys

import auth


def _ask(prompt: str) -> str:
    value = getpass.getpass(prompt)
    if not value:
        print("Nothing entered.")
        raise SystemExit(1)
    return value


def cmd_list() -> None:
    everyone = auth.users()
    if not everyone:
        print("No accounts. The app runs open, on its original workspace.")
        return
    print(f"{len(everyone)} account(s):")
    for name in sorted(everyone):
        home = auth.workspace_for(name)
        note = "  (the shared workspace)" if home == auth.HERE else ""
        print(f"  {name:20s} {home}{note}")


def cmd_add(name: str) -> None:
    password = _ask(f"Password for {name}: ")
    if password != _ask("Again: "):
        print("Those did not match.")
        raise SystemExit(1)
    try:
        auth.register(name, password)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)
    print(f"Added {name}. Workspace: {auth.workspace_for(name)}")


def cmd_passwd(name: str) -> None:
    if name not in auth.users():
        print(f"There is no account called {name}.")
        raise SystemExit(1)
    password = _ask(f"New password for {name}: ")
    if password != _ask("Again: "):
        print("Those did not match.")
        raise SystemExit(1)
    if len(password) < 8:
        print("Use at least 8 characters.")
        raise SystemExit(1)
    # set it directly: the point of this command is not knowing the old one
    import secrets

    everyone = auth.users()
    salt = secrets.token_bytes(16)
    everyone[name]["salt"] = salt.hex()
    everyone[name]["hash"] = auth._hash(password, salt)
    auth._write(auth.USERS, everyone)
    live = {t: s for t, s in auth._read(auth.SESSIONS, {}).items()
            if s.get("user") != name}
    auth._write(auth.SESSIONS, live)
    print(f"Password changed for {name}. Their other sessions were signed out.")


def cmd_remove(name: str, purge: bool) -> None:
    everyone = auth.users()
    if name not in everyone:
        print(f"There is no account called {name}.")
        raise SystemExit(1)
    home = auth.workspace_for(name)
    everyone.pop(name)
    auth._write(auth.USERS, everyone)
    live = {t: s for t, s in auth._read(auth.SESSIONS, {}).items()
            if s.get("user") != name}
    auth._write(auth.SESSIONS, live)

    if purge and home != auth.HERE:
        shutil.rmtree(home, ignore_errors=True)
        print(f"Removed {name} and deleted {home}.")
    else:
        where = "the shared workspace" if home == auth.HERE else str(home)
        print(f"Removed {name}. Their work is still in {where}.")
    if not everyone:
        print("No accounts left, so the app runs open again.")


def cmd_off() -> None:
    everyone = auth.users()
    if not everyone:
        print("There were no accounts.")
        return
    auth._write(auth.USERS, {})
    auth._write(auth.SESSIONS, {})
    print(f"Removed {len(everyone)} account(s). Every workspace is still on "
          f"disk under {auth.DATA / 'users'}; the app now opens the shared one.")


def main(argv: list) -> None:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    command, rest = argv[0], argv[1:]
    if command == "list":
        cmd_list()
    elif command == "add" and rest:
        cmd_add(rest[0].strip().lower())
    elif command == "passwd" and rest:
        cmd_passwd(rest[0].strip().lower())
    elif command == "remove" and rest:
        cmd_remove(rest[0].strip().lower(), "--purge" in rest)
    elif command == "off":
        cmd_off()
    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
