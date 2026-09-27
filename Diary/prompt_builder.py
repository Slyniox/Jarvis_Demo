###############################################################################

from memory_retriever import MemoryRetriever

retriever = MemoryRetriever()

###############################################################################


def build_memory_context(queries, top_k, max_full_entries, max_summaries):

    result = retriever.retrieve_multiple(queries, top_k, max_full_entries, max_summaries)

    if (
        not result["profiles"]
        and not result["full_entries"]
        and not result["summaries"]
    ):
        return ""

    lines = []

    lines.append("""
    ===================================================
    LONG-TERM MEMORY
    ===================================================

    The following information comes from the user's
    personal diary and long-term memory.
    Use it if it is relevant.
    Ignore it if it is unrelated to the current request.
""")
    ###########################################################################
    # Person profiles
    ###########################################################################

    if result["profiles"]:

        lines.append("PERSON PROFILES")
        lines.append("----------------")

        for profile in result["profiles"]:

            lines.append("")
            lines.append(f"## {profile['name']}")
            lines.append(profile["content"])
            lines.append("")

    ###########################################################################
    # Full diary entries
    ###########################################################################

    if result["full_entries"]:

        lines.append("")
        lines.append("MOST RELEVANT DIARY ENTRIES")
        lines.append("--------------------------")

        for entry in result["full_entries"]:

            lines.append("")
            lines.append(f"Date: {entry['date']}")
            lines.append("")
            lines.append(entry["entry"])
            lines.append("")
            lines.append("-" * 60)

    ###########################################################################
    # Summaries
    ###########################################################################

    if result["summaries"]:

        lines.append("")
        lines.append("RELATED MEMORIES")
        lines.append("----------------")

        for memory in result["summaries"]:

            metadata = memory.get("metadata") or {}

            lines.append("")
            lines.append(
                f"Date: {memory['date']}"
            )

            ###################################################################
            # Summary
            ###################################################################

            summary = metadata.get("summary")

            if summary:
                lines.append(
                    f"Summary: {summary}"
                )

            ###################################################################
            # People
            ###################################################################

            people = metadata.get("people", [])

            if people:
                lines.append(
                    "People: " + ", ".join(people)
                )

            ###################################################################
            # Places
            ###################################################################

            places = metadata.get("places", [])

            if places:
                lines.append(
                    "Places: " + ", ".join(places)
                )

            ###################################################################
            # Projects
            ###################################################################

            projects = metadata.get("projects", [])

            if projects:
                lines.append(
                    "Projects: " + ", ".join(projects)
                )

            ###################################################################
            # Companies
            ###################################################################

            companies = metadata.get("companies", [])

            if companies:
                lines.append(
                    "Companies: " + ", ".join(companies)
                )

            ###################################################################
            # Goals
            ###################################################################

            goals = metadata.get("goals", [])

            if goals:
                lines.append(
                    "Goals: " + ", ".join(goals)
                )

            ###################################################################
            # Emotions
            ###################################################################

            emotions = metadata.get("emotions", [])

            if emotions:
                lines.append(
                    "Emotions: " + ", ".join(emotions)
                )

            ###################################################################
            # Events
            ###################################################################

            events = metadata.get("events", [])

            if events:

                lines.append("Events:")

                for event in events:
                    lines.append(
                        f"  • {event}"
                    )

            lines.append("")
            lines.append("-" * 60)

    ###########################################################################
    # Return final context
    ###########################################################################

    return "\n".join(lines)