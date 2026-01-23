// Simple markdown parser
function parseMarkdown(markdown) {
    let html = markdown;
    
    // Escape HTML special characters first to prevent XSS
    // (but we'll skip this for now to allow embedding HTML if needed)
    
    // Process code blocks first (before other processing)
    const codeBlocks = [];
    html = html.replace(/```([\s\S]*?)```/gim, (match, code) => {
        const placeholder = `___CODE_BLOCK_${codeBlocks.length}___`;
        codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
        return placeholder;
    });
    
    // Inline code (before other processing)
    const inlineCodes = [];
    html = html.replace(/`([^`]+)`/gim, (match, code) => {
        const placeholder = `___INLINE_CODE_${inlineCodes.length}___`;
        inlineCodes.push(`<code>${code}</code>`);
        return placeholder;
    });
    
    // Headers (must be on their own line)
    html = html.replace(/^### (.*)$/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*)$/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*)$/gim, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^\)]+)\)/gim, '<a href="$2">$1</a>');
    
    // Process lists (unordered)
    html = html.replace(/^((?:\s*-\s+.+$\n?)+)/gim, (match) => {
        const items = match.trim().split('\n').filter(line => line.trim()).map(line => {
            const content = line.replace(/^\s*-\s+/, '').trim();
            return `<li>${content}</li>`;
        }).join('');
        return `<ul>${items}</ul>`;
    });
    
    // Ordered lists
    html = html.replace(/^((?:\s*\d+\.\s+.+$\n?)+)/gim, (match) => {
        const items = match.trim().split('\n').filter(line => line.trim()).map(line => {
            const content = line.replace(/^\s*\d+\.\s+/, '').trim();
            return `<li>${content}</li>`;
        }).join('');
        return `<ol>${items}</ol>`;
    });
    
    // Split into paragraphs by double newlines
    const blocks = html.split(/\n\n+/);
    html = blocks.map(block => {
        block = block.trim();
        if (!block) return '';
        
        // Don't wrap if it's already a block element
        if (block.match(/^<(h[1-6]|ul|ol|pre|blockquote|div|hr)/)) {
            return block;
        }
        
        // Single line breaks within paragraphs become <br>
        block = block.replace(/\n/g, '<br>');
        
        return `<p>${block}</p>`;
    }).filter(b => b).join('\n');
    
    // Restore code blocks
    codeBlocks.forEach((code, i) => {
        html = html.replace(`___CODE_BLOCK_${i}___`, code);
    });
    
    // Restore inline code
    inlineCodes.forEach((code, i) => {
        html = html.replace(`___INLINE_CODE_${i}___`, code);
    });
    
    return html;
}
