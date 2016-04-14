grades = [];
level = 5;
semester = 'Fall';
subject = 'Math';
student = 'John_Doe';
fieldnames = {semester subject student}
newGrades_Doe = [85, 89, 76, 93, 85, 91, 68, 84, 95, 73];

grades = setfield(grades, {level}, ...
                  fieldnames{:}, {10, 21:30}, ... 
                  newGrades_Doe);
              

grades = setfield(grades, {2}, ...
  'Fall', {1}, ...
  4);

              
grades(5).Fall(10,21:30) = newGrades_Doe

% View the new contents.
grades(level).(semester).(subject).(student)(10, 21:30)


setfield(all_data, {1:length(all_data)}, 'wrongInitiation', 1)

all_data = setfield(all_data, {20}, 'wrongInitiation', [4])


grades = setfield(grades, {5}, ...
                  'Fall', {10, 21:30}, ... 
                  newGrades_Doe);
              
        

a.b = 0;

a = arrayfun(@(x) setfield(x, 'c', [4]), a)

%%
[all_data.didNotInitiate] = all_data.wrongInitiation;
[all_data(3:4).wrongInitiation] = deal(31,15)

all_data = setfield(all_data, {20}, 'wrongInitiation', [4]) % but you cannot use this for several elements.
