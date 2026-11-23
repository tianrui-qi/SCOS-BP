import hydra
import omegaconf
import base64
import json
import os
import pathlib
import tempfile
import tqdm
import urllib.parse
import urllib.request


RESOLVE_URL = (
    "https://ckdatabasews.icloud.com/database/1/"
    "com.apple.cloudkit/production/public/records/resolve"
)


def resolve(link: str, filename: str) -> tuple[str, int]:
    share_id = urllib.parse.urlsplit(link).path.rstrip("/").rsplit("/", 1)[-1]
    request = urllib.request.Request(
        RESOLVE_URL,
        data=json.dumps({"shortGUIDs": [{"value": share_id}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)["results"][0]
    record = result.get("rootRecord", {})
    if result.get("requireAppleLogin") or record.get("recordType") != "content":
        raise ValueError("Expected a public iCloud file link")

    fields = record["fields"]
    name = base64.b64decode(fields["encryptedBasename"]["value"]).decode()
    extension = fields.get("extension", {}).get("value")
    if extension:
        name += f".{extension}"
    if name != filename:
        raise ValueError(f"Link shares {name!r}, expected {filename!r}")
    asset = fields["fileContent"]["value"]
    url = asset["downloadURL"].replace(
        "${f}", urllib.parse.quote(name, safe="")
    )
    return url, int(asset["size"])


def download(link: str, path: pathlib.Path) -> None:
    url, size = resolve(link, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        temporary = pathlib.Path(folder) / path.name
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            with tqdm.tqdm(
                total=size, unit="B", unit_scale=True, desc=path.name
            ) as bar:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    bar.update(len(chunk))
        if temporary.stat().st_size != size:
            raise OSError(f"Incomplete download (expected {size:,} bytes)")
        # Publish the complete file without replacing an existing checkpoint.
        os.link(temporary, path)


@hydra.main(
    version_base=None, config_path="../config",
    config_name="pipeline/setup"
)
def main(cfg: omegaconf.DictConfig) -> None:
    failed = []
    for file in cfg.files:
        path = pathlib.Path(hydra.utils.to_absolute_path(file.path))
        if path.is_file():
            print(f"Already exists: {file.path}")
            continue
        try:
            download(file.link, path)
            print(f"Downloaded: {file.path}")
        except Exception as error:
            failed.append(f"{file.path}: {error}")
    if failed:
        raise RuntimeError("Setup failed for:\n" + "\n".join(failed))


if __name__ == "__main__": main()
