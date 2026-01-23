// Simple markdown parser
function parseMarkdown(markdown) {
    let html = markdown;
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^\)]+)\)/gim, '<a href="$2">$1</a>');
    
    // Lists - Unordered
    html = html.replace(/^\s*-\s+(.*)$/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // Lists - Ordered
    html = html.replace(/^\s*\d+\.\s+(.*)$/gim, '<li>$1</li>');
    
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');
    
    // Paragraphs - split by double newlines
    let paragraphs = html.split(/\n\n+/);
    html = paragraphs.map(p => {
        p = p.trim();
        // Don't wrap if already a tag
        if (p.match(/^<(h1|h2|h3|h4|ul|ol|pre|li)/)) {
            return p;
        }
        return `<p>${p}</p>`;
    }).join('\n');
    
    // Line breaks
    html = html.replace(/\n/gim, '<br>');
    
    // Clean up list formatting
    html = html.replace(/<br><ul>/gim, '<ul>');
    html = html.replace(/<\/ul><br>/gim, '</ul>');
    html = html.replace(/<br><ol>/gim, '<ol>');
    html = html.replace(/<\/ol><br>/gim, '</ol>');
    html = html.replace(/<br><\/li>/gim, '</li>');
    html = html.replace(/<li><br>/gim, '<li>');
    
    // Clean up header formatting
    html = html.replace(/<br><\/h([123])>/gim, '</h$1>');
    html = html.replace(/<h([123])><br>/gim, '<h$1>');
    
    // Clean up paragraph formatting
    html = html.replace(/<p><br>/gim, '<p>');
    html = html.replace(/<br><\/p>/gim, '</p>');
    
    return html;
}
