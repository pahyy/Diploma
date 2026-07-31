class ResponseFormatter:

    def format(self, user_question: str, params: dict, aggregated: dict,
               summary: str, chart_path, validation: dict) -> dict:
        response = {
            "question": user_question,
            "summary": summary,
            "chart_path": chart_path,
            "analyzers_used": aggregated["analyzers_used"],
            "analysis": aggregated["analysis"],
            "supporting": aggregated["supporting"],
            "validation": validation,
            "params": params,
        }

        if validation and not validation["valid"]:
            response["warning"] = (
                "Opozorilo: nekaterih številk v povzetku ni bilo mogoče preveriti "
                f"v rezultatih analize: {', '.join(validation['unverified'])}"
            )

        return response

    def to_text(self, response: dict) -> str:
        lines = ["", "=" * 60, response["summary"]]
        if response.get("warning"):
            lines += ["", response["warning"]]
        if response.get("chart_path"):
            lines += ["", f"Graf shranjen v: {response['chart_path']}"]
        lines.append("=" * 60)
        return "\n".join(lines)