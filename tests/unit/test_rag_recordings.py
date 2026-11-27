import httpx
import pytest

from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR, LitSettings
from mcp_servers.lit.http import Recordings, open_http
from mcp_servers.lit.requests import corpus_pubmed_search_url
from rag import record_corpus
from rag.config import CORPUS_RECORDINGS_DIR


def test_shared_recordings_match_the_eval_recordings_byte_for_byte(tmp_path):
    source = Recordings(DEFAULT_RECORDINGS_DIR)
    committed = Recordings(CORPUS_RECORDINGS_DIR)
    copied = Recordings(tmp_path)
    urls = record_corpus.shared_recording_urls()
    assert len(set(urls)) == 5

    record_corpus.copy_shared_recordings(tmp_path)

    for url in urls:
        expected = source.path_for("GET", url).read_bytes()
        assert committed.path_for("GET", url).read_bytes() == expected
        assert copied.path_for("GET", url).read_bytes() == expected


async def test_pubmed_recording_fetches_batches_of_fifty_in_search_order(tmp_path):
    pmids = [str(i) for i in range(201, 149, -1)]
    fetched = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            assert dict(request.url.params) == {
                "db": "pubmed",
                "term": '"KRAS G12C"[Title/Abstract] AND (sotorasib OR adagrasib OR "AMG 510" OR MRTX849) AND hasabstract',
                "retmode": "json",
                "retmax": "200",
                "sort": "relevance",
            }
            return httpx.Response(
                200, json={"esearchresult": {"count": "52", "idlist": pmids}}
            )

        assert request.url.path.endswith("/efetch.fcgi")
        assert request.url.params["retmode"] == "xml"
        batch = request.url.params["id"].split(",")
        fetched.append(batch)
        rows = "".join(
            f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID></MedlineCitation></PubmedArticle>"
            for pmid in reversed(batch)
        )
        return httpx.Response(200, text=f"<PubmedArticleSet>{rows}</PubmedArticleSet>")

    async with open_http(
        LitSettings(mode="record", recordings_dir=tmp_path),
        transport=httpx.MockTransport(respond),
    ) as http:
        recorded = await record_corpus.fetch_pubmed_corpus(http)

    assert fetched == [pmids[:50], pmids[50:]]
    assert [row.pmid for row in recorded] == pmids
    assert Recordings(tmp_path).contains("GET", corpus_pubmed_search_url())

    async with open_http(LitSettings(mode="replay", recordings_dir=tmp_path)) as http:
        assert await record_corpus.fetch_pubmed_corpus(http) == recorded


@pytest.mark.parametrize(("mode", "key"), [("replay", None), ("record", "fixture-key")])
def test_recording_command_requires_record_mode_without_an_api_key(
    mode, key, monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", ["record_corpus"])
    monkeypatch.setenv("LIT_MCP_MODE", mode)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("NCBI_API_KEY", key)

    with pytest.raises(SystemExit, match="2"):
        record_corpus.main()
    assert "Set LIT_MCP_MODE=record and unset NCBI_API_KEY" in capsys.readouterr().err


def test_recording_command_uses_the_corpus_directory(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["record_corpus"])
    monkeypatch.setenv("LIT_MCP_MODE", "record")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", str(tmp_path))

    async def record(settings: LitSettings) -> None:
        assert settings.mode == "record"
        assert settings.ncbi_api_key is None
        assert settings.recordings_dir == CORPUS_RECORDINGS_DIR

    monkeypatch.setattr(record_corpus, "_record", record)
    record_corpus.main()
