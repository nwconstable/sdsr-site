# sdsr-site

A simple website for graduate course assignments, featuring team member profiles and dynamic markdown-based assignment submissions.

## Features

- **Home Page**: Landing page with biographical sections for two group members
  - Profile pictures (placeholder included)
  - Names, emails, and bio information
  - Dropdown navigation to assignments
  
- **Assignment Pages**: Dynamic assignment viewer that loads markdown files
  - Assignments written in markdown format
  - Dropdown menu to select assignments
  - Automatic rendering of markdown content
  - Theme support (dark/light mode)

## Usage

### Viewing the Website Locally

You need to run a local web server to view the website properly (due to JavaScript fetch requirements):

```bash
# Using Python 3
python3 -m http.server 8000

# Using Python 2
python -m SimpleHTTPServer 8000

# Using Node.js (if you have http-server installed)
npx http-server
```

Then open `http://localhost:8000` in your web browser.

### Customizing the Content

1. **Update Team Member Information**:
   - Edit `index.html` and update the team member names, emails, and bio sections
   - Replace placeholder images in the `images/` directory with actual profile photos (named `member1.jpg` and `member2.jpg`)

2. **Add New Assignments**:
   - Create a new markdown file in the `assignments/` directory (e.g., `assignment3.md`)
   - Edit `assignments-config.json` to add the new assignment:
     ```json
     {
       "id": "assignment3",
       "title": "Assignment 3: Title Here",
       "file": "assignments/assignment3.md"
     }
     ```
   - The dropdown menu will automatically update with the new assignment

3. **Edit Assignment Content**:
   - Edit the markdown files in the `assignments/` directory
   - Use standard markdown syntax for formatting

4. **Customize Styling**:
   - Edit `styles.css` to change colors, fonts, or layout
   - The site includes dark mode (default) and light mode themes

## File Structure

```
sdsr-site/
├── index.html              # Home page with team member bios
├── assignment.html         # Assignment viewer page
├── assignments/            # Directory for assignment markdown files
│   ├── assignment1.md      # First assignment
│   └── assignment2.md      # Second assignment
├── assignments-config.json # Configuration file listing all assignments
├── assignment-loader.js    # JavaScript for loading and rendering assignments
├── theme-toggle.js         # JavaScript for theme switching
├── styles.css             # Main stylesheet
├── images/                # Directory for images
│   └── placeholder.svg    # Placeholder profile image
└── README.md              # This file
```

## Assignment Markdown Format

Assignments should be written in markdown format. Example:

```markdown
# Assignment Title

**Due Date:** Date here | **Submitted by:** Names here

## Section 1

Content here...

## Section 2

More content...
```

## Theme Support

The website includes dark mode (default) and light mode:
- Click the theme toggle button in the navigation bar to switch
- Theme preference is saved in localStorage
- Works across all pages

## Deployment

To deploy this website:

1. **GitHub Pages**: Push to GitHub and enable GitHub Pages in repository settings
2. **Other Hosting**: Upload all files to any web hosting service that supports static sites

## License

This project is for educational purposes.