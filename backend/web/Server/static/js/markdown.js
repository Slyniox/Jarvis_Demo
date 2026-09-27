// Lightweight, dependency-free Markdown renderer with common GFM features.
// Raw HTML is escaped before rendering.

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function splitTableRow(line) {
  let value = line.trim();

  if (value.startsWith("|")) value = value.slice(1);
  if (value.endsWith("|") && !value.endsWith("\\|")) value = value.slice(0, -1);

  const cells = [];
  let current = "";
  let escaped = false;

  for (const char of value) {
    if (char === "|" && !escaped) {
      cells.push(current.trim());
      current = "";
      continue;
    }

    if (char === "\\" && !escaped) {
      escaped = true;
      current += char;
      continue;
    }

    escaped = false;
    current += char;
  }

  cells.push(current.trim());
  return cells;
}

function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return cells.length >= 1 &&
    cells.every(cell => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")));
}

function tableAlignment(cell) {
  const value = cell.replace(/\s/g, "");
  if (value.startsWith(":") && value.endsWith(":")) return "center";
  if (value.endsWith(":")) return "right";
  if (value.startsWith(":")) return "left";
  return "";
}

function renderTable(lines) {
  const headers = splitTableRow(lines[0]);
  const separator = splitTableRow(lines[1]);
  const columnCount = Math.max(headers.length, separator.length);
  const normalized = cells =>
    Array.from({ length: columnCount }, (_, i) => cells[i] ?? "");

  let html = '<div class="table-wrap"><table><thead><tr>';

  normalized(headers).forEach((cell, i) => {
    const align = tableAlignment(separator[i] ?? "");
    const attr = align ? ` style="text-align:${align}"` : "";
    html += `<th${attr}>${inlineMarkdown(cell)}</th>`;
  });

  html += "</tr></thead><tbody>";

  for (let i = 2; i < lines.length; i++) {
    const cells = normalized(splitTableRow(lines[i]));
    html += "<tr>";
    cells.forEach((cell, index) => {
      const align = tableAlignment(separator[index] ?? "");
      const attr = align ? ` style="text-align:${align}"` : "";
      html += `<td${attr}>${inlineMarkdown(cell)}</td>`;
    });
    html += "</tr>";
  }

  html += "</tbody></table></div>";
  return html;
}

function inlineMarkdown(text) {
  let escaped = escapeHtml(text);

  const code = [];
  escaped = escaped.replace(/`([^`\n]+)`/g, (_, value) => {
    code.push(`<code>${value}</code>`);
    return `\u0000CODE${code.length - 1}\u0000`;
  });

  escaped = escaped.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  escaped = escaped.replace(
    /(^|[\s(])(https?:\/\/[^\s<]+)(?=$|[\s).,!?:;])/g,
    '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
  );

  escaped = escaped.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  escaped = escaped.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
  escaped = escaped.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
  escaped = escaped.replace(/(?<!_)_([^_\n]+)_(?!_)/g, "<em>$1</em>");
  escaped = escaped.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");

  escaped = escaped.replace(/\\\|/g, "|");
  escaped = escaped.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => code[index]);
  return escaped;
}

function listMarker(line) {
  const match = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
  if (!match) return null;

  const indent = match[1].replace(/\t/g, "    ").length;
  const marker = match[2];
  return {
    indent,
    ordered: /^\d/.test(marker),
    content: match[3],
  };
}

// Render a list recursively based on indentation. This handles structures such as:
//
// - One
//   - Two
//     - Three
//   - Four
// - Five
//
// and mixed ordered/unordered nesting.
function renderList(lines, start, baseIndent, ordered) {
  const tag = ordered ? "ol" : "ul";
  let html = `<${tag}>`;
  let i = start;

  while (i < lines.length) {
    const marker = listMarker(lines[i]);

    if (!marker || marker.indent !== baseIndent || marker.ordered !== ordered) {
      break;
    }

    html += `<li>${inlineMarkdown(marker.content)}`;

    const continuation = [];
    let nestedStart = -1;
    let nestedIndent = null;

    i++;

    while (i < lines.length) {
      const next = listMarker(lines[i]);

      if (!next) {
        if (!lines[i].trim()) {
          continuation.push("");
          i++;
          continue;
        }

        const rawIndent = lines[i].match(/^\s*/)[0].replace(/\t/g, "    ").length;
        if (rawIndent > baseIndent) {
          continuation.push(lines[i].trim());
          i++;
          continue;
        }

        break;
      }

      if (next.indent <= baseIndent) {
        break;
      }

      // A deeper list marker belongs to this item.
      nestedStart = i;
      nestedIndent = next.indent;
      break;
    }

    if (continuation.length) {
      const text = continuation.filter(Boolean).map(inlineMarkdown).join("<br>");
      if (text) html += `<div class="list-continuation">${text}</div>`;
    }

    if (nestedStart !== -1) {
      const nested = renderList(lines, nestedStart, nestedIndent, listMarker(lines[nestedStart]).ordered);
      html += nested.html;
      i = nested.nextIndex;
    }

    html += "</li>";
  }

  html += `</${tag}>`;
  return { html, nextIndex: i };
}

function isListLine(line) {
  return !!listMarker(line);
}

function renderBlock(block) {
  const lines = block.split("\n");
  let html = "";
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code blocks. Support fences longer than three backticks so
    // Markdown examples containing nested ``` blocks render correctly.
    const fence = line.match(/^(`{3,})([A-Za-z0-9_+#.-]*)\s*$/);
    if (fence) {
      const fenceLength = fence[1].length;
      const language = fence[2];
      const codeLines = [];
      i++;

      while (i < lines.length) {
        const candidate = lines[i].trim();
        const closingFence = candidate.match(/^(`{3,})\s*$/);

        if (closingFence && closingFence[1].length >= fenceLength) {
          i++;
          break;
        }

        codeLines.push(lines[i]);
        i++;
      }

      const lang = language
        ? `<span class="code-language">${escapeHtml(language)}</span>`
        : "";

      html += `<div class="code-block">${lang}<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre></div>`;
      continue;
    }

    if (
      i + 1 < lines.length &&
      line.includes("|") &&
      isTableSeparator(lines[i + 1])
    ) {
      const tableLines = [line, lines[i + 1]];
      i += 2;

      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        tableLines.push(lines[i]);
        i++;
      }

      html += renderTable(tableLines);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
      i++;
      continue;
    }

    if (/^\s*(---+|\*\*\*+)\s*$/.test(line)) {
      html += "<hr>";
      i++;
      continue;
    }

    const marker = listMarker(line);
    if (marker) {
      const result = renderList(lines, i, marker.indent, marker.ordered);
      html += result.html;
      i = result.nextIndex;
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quote.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      html += `<blockquote>${renderMarkdown(quote.join("\n"))}</blockquote>`;
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    const paragraph = [line];
    i++;

    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^```/.test(lines[i]) &&
      !/^#{1,6}\s+/.test(lines[i]) &&
      !isListLine(lines[i]) &&
      !/^\s*>\s?/.test(lines[i]) &&
      !/^\s*(---+|\*\*\*+)\s*$/.test(lines[i]) &&
      !(
        i + 1 < lines.length &&
        lines[i].includes("|") &&
        isTableSeparator(lines[i + 1])
      )
    ) {
      paragraph.push(lines[i]);
      i++;
    }

    html += `<p>${inlineMarkdown(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`;
  }

  return html;
}

export function renderMarkdown(text) {
  return renderBlock(String(text ?? ""));
}
