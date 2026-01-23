# sdsr-site

A simple website for graduate course assignments, featuring team member profiles and assignment submissions.

## Features

- **Home Page**: Landing page with biographical sections for two group members
  - Profile pictures (placeholder included)
  - Names, emails, and bio information
  - Navigation to assignments
  
- **Assignment Pages**: Individual pages for each assignment submission
  - Formatted content sections
  - Easy navigation between home and assignments

## Usage

### Viewing the Website Locally

Simply open `index.html` in your web browser to view the website locally.

### Customizing the Content

1. **Update Team Member Information**:
   - Edit `index.html` and update the team member names, emails, and bio sections
   - Replace placeholder images in the `images/` directory with actual profile photos (named `member1.jpg` and `member2.jpg`)

2. **Add New Assignments**:
   - Copy `assignment1.html` and create `assignment2.html`, `assignment3.html`, etc.
   - Update the content in each assignment file
   - Add navigation links in both `index.html` and the new assignment pages

3. **Customize Styling**:
   - Edit `styles.css` to change colors, fonts, or layout

## File Structure

```
sdsr-site/
├── index.html           # Home page with team member bios
├── assignment1.html     # First assignment page
├── styles.css          # Main stylesheet
├── images/             # Directory for images
│   └── placeholder.svg # Placeholder profile image
└── README.md           # This file
```

## Deployment

To deploy this website:

1. **GitHub Pages**: Push to GitHub and enable GitHub Pages in repository settings
2. **Other Hosting**: Upload all files to any web hosting service

## License

This project is for educational purposes.