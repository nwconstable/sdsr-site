// Assignment loader functionality
document.addEventListener('DOMContentLoaded', async () => {
    const assignmentContent = document.getElementById('assignment-content');
    const assignmentsMenu = document.getElementById('assignments-menu');
    const dropdownToggle = document.getElementById('assignments-dropdown');

    // Load assignments configuration
    let assignmentsConfig = [];
    try {
        const response = await fetch('assignments-config.json');
        const config = await response.json();
        assignmentsConfig = config.assignments;
    } catch (error) {
        console.error('Failed to load assignments configuration:', error);
        if (assignmentsMenu) {
            assignmentsMenu.innerHTML = '<li class="dropdown-error">No assignments available</li>';
        }
        return;
    }

    // Populate dropdown menu
    if (assignmentsMenu && assignmentsConfig.length > 0) {
        assignmentsMenu.innerHTML = '';
        assignmentsConfig.forEach(assignment => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `assignment.html?id=${assignment.id}`;
            a.textContent = assignment.title;
            a.classList.add('dropdown-item');
            li.appendChild(a);
            assignmentsMenu.appendChild(li);
        });
    }

    // Toggle dropdown on click
    if (dropdownToggle) {
        dropdownToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const dropdown = dropdownToggle.closest('.dropdown');
            dropdown.classList.toggle('active');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            const dropdown = dropdownToggle.closest('.dropdown');
            if (dropdown) {
                dropdown.classList.remove('active');
            }
        });
    }

    // Load specific assignment if on assignment.html page
    if (assignmentContent && window.location.pathname.includes('assignment.html')) {
        const urlParams = new URLSearchParams(window.location.search);
        const assignmentId = urlParams.get('id');

        if (!assignmentId) {
            assignmentContent.innerHTML = `
                <div class="error-message">
                    <h2>No Assignment Selected</h2>
                    <p>Please select an assignment from the dropdown menu above.</p>
                </div>
            `;
            return;
        }

        // Find the assignment in config
        const assignment = assignmentsConfig.find(a => a.id === assignmentId);
        if (!assignment) {
            assignmentContent.innerHTML = `
                <div class="error-message">
                    <h2>Assignment Not Found</h2>
                    <p>The requested assignment could not be found.</p>
                </div>
            `;
            return;
        }

        // Update page title
        document.title = assignment.title + ' - SDSR Course';

        // Load and render markdown
        try {
            const response = await fetch(assignment.file);
            if (!response.ok) {
                throw new Error('Failed to load assignment file');
            }
            const markdown = await response.text();
            
            // Use our custom markdown parser
            const html = parseMarkdown(markdown);
            assignmentContent.innerHTML = `<div class="assignment-content">${html}</div>`;
        } catch (error) {
            console.error('Failed to load assignment:', error);
            assignmentContent.innerHTML = `
                <div class="error-message">
                    <h2>Error Loading Assignment</h2>
                    <p>Failed to load the assignment content. Please try again later.</p>
                </div>
            `;
        }
    }
});
